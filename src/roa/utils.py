import contextlib
import pickle
import warnings

import numpy as np
from pyresample.geometry import AreaDefinition
from pyresample.kd_tree import resample_nearest
try:
    from satpy.area import get_area_def
except:
    from satpy.resample import get_area_def
    warnings.warn(
        "Falling back to legacy 'satpy.resample.get_area_def'. "
        "Consider upgrading Satpy; this path is deprecated and slated for removal.",
    )
from scipy.stats import loguniform
import torch
import xarray as xr

from roa.data import SEVIRI_0DEG_AREADEF

def fci2seviri_generate_nearest_neighbour_linesample_arrays(
    output_file
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Generate the indexes that map FCI to SEVIRI grid

    Args:
        output_file: where to save the generated arrays.

    Note:
        Reading the native SEVIRI data results in flipped coordinates than
        what SEVIRI_0DEG_AREADEF expects, and reading the native FCI data results
        in coordinates flipped upside down compared to what a "FCI_0DEG_AREADEF"
        would expect.

        That is, when reading the native data, the corresponding coordinates are (close to)

        x_seviri = SEVIRI_0DEG_AREADEF.projection_x_coords[::-1]
        y_seviri = SEVIRI_0DEG_AREADEF.projection_y_coords[::-1]

        x_fci = FCI_0DEG_AREADEF.projection_x_coords
        y_fci = FCI_0DEG_AREADEF.projection_y_coords[::-1]

        This function generates the arrays for the nearest neighbour applied upon
        reading the data, not on "SEVIRI/FCI_0DEG_AREADEF" area definitions. That is,
        it matches the {x,y}_{seviri,fci} definitions above.
    """
    fci_0deg_areadef = get_area_def('mtg_fci_fdss_2km')

    fci_0deg_areadef = AreaDefinition(
        fci_0deg_areadef.area_id,
        fci_0deg_areadef.description,
        fci_0deg_areadef.proj_id,
        fci_0deg_areadef.crs,
        fci_0deg_areadef.width,
        fci_0deg_areadef.height,
        tuple(np.array(fci_0deg_areadef.area_extent)[[0, 3, 2, 1]].tolist())
    )

    seviri_0deg_areadef = AreaDefinition(
        SEVIRI_0DEG_AREADEF.area_id,
        SEVIRI_0DEG_AREADEF.description,
        SEVIRI_0DEG_AREADEF.proj_id,
        SEVIRI_0DEG_AREADEF.crs,
        SEVIRI_0DEG_AREADEF.width,
        SEVIRI_0DEG_AREADEF.height,
        tuple(np.array(SEVIRI_0DEG_AREADEF.area_extent)[[2, 3, 0, 1]].tolist())
    )

    idx_col, idx_row = np.meshgrid(
        np.arange(fci_0deg_areadef.width),
        np.arange(fci_0deg_areadef.height)
    )
    idx_col_map = resample_nearest(
        fci_0deg_areadef,
        idx_col,
        seviri_0deg_areadef,
        radius_of_influence=3e3,
        fill_value=-1
    )
    idx_row_map = resample_nearest(
        fci_0deg_areadef,
        idx_row,
        seviri_0deg_areadef,
        radius_of_influence=3e3,
        fill_value=-1
    )
    idx_col_map = xr.DataArray(
        idx_col_map,
        dims=['y', 'x'],
        coords={
            'y': seviri_0deg_areadef.projection_y_coords,
            'x': seviri_0deg_areadef.projection_x_coords
        }
    )
    idx_row_map = xr.DataArray(
        idx_row_map,
        dims=['y', 'x'],
        coords={
            'y': seviri_0deg_areadef.projection_y_coords,
            'x': seviri_0deg_areadef.projection_x_coords
        }
    )
    mask_invalid = (idx_col_map == -1) | (idx_row_map == -1)

    if output_file is not None:
        with open(output_file, 'wb') as handle:
            pickle.dump((idx_row_map, idx_col_map, mask_invalid), handle)

    return idx_row_map, idx_col_map, mask_invalid


@contextlib.contextmanager
def temp_seed(seed: int):
    """
    Temporarily set the random seed for the duration of the context.
    https://stackoverflow.com/a/49557127
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def categorizer(x: np.ndarray[float],
                thresholds: np.ndarray[float]=np.array([0.2, 2.5, 10, 50])
    ) -> np.ndarray[float]:
    """
    Categorize the input array `x` into bins defined by the `thresholds` array.

    Args:
        x: The input array to categorize.
        thresholds: The thresholds to use to categorize the input array.

    Returns:
        The categorized array, where each element is the index of the bin in which
        the corresponding element of the input array falls, according to
        numpy.digitize.
    """
    assert len(thresholds) > 1
    x = np.digitize(x, thresholds) * np.where(np.isfinite(x), 1, np.nan)
    return x

class VaryZeros:
    """Change zeros with a random number between `a` and `b` sampled from a
    loguniform distribution."""

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, sample):
        assert isinstance(sample, torch.Tensor)

        # Mask where it is zero
        mask = torch.isclose(sample,
            torch.zeros(sample.shape, dtype=sample.dtype, device=sample.device))
        n = mask.sum().item()
        sample[mask] = torch.from_numpy(loguniform.rvs(self.a, self.b, size=n)).type(sample.type()).to(sample.device)
        return sample
    
    def invert(self, sample):
        assert isinstance(sample, torch.Tensor)
        mask = sample <= self.b
        sample[mask] = 0
        return sample

class VaryZerosLog(VaryZeros):
    """Apply a log transform to the VaryZeros transform"""
    def __init__(self, a=1e-3, b=1e-2):
        super().__init__(a, b)
    
    def __call__(self, sample):
        return torch.log(super().__call__(sample))

    def invert(self, sample):
        return super().invert(torch.exp(sample))
    
def mask_invalid_rates(
    a: np.ndarray,
    max_rate: float = 1e2,
):
    """
    Occasionally, the precipitation rates
    can be considered invalid. This can happen
    due to a variety of reasons.
    """
    return np.where(
        np.isfinite(a),
        np.where(
            (a >= 0) & (a < max_rate),
            a,
            np.nan
        ),
        np.nan
    )
