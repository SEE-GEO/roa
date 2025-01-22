from pathlib import Path
import warnings

import numpy as np
from numba import jit
from pansat.products.satellite.gpm import l2b_gpm_cmb
from pyproj import Transformer
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.kd_tree import resample_nearest
from satpy import Scene
from satpy.resample import get_area_def
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import xarray as xr

QUANTILES = np.arange(1, 100) / 100

CHANNELS_MAP = {
    5: 'WV_062',
    6: 'WV_073',
    7: 'IR_087',
    8: 'IR_097',
    9: 'IR_108',
    10: 'IR_120',
    11: 'IR_134'
}

TRAINING_SET_STATISTICS = {
    "mean": {
        "IR_087": 282.1137428513001,
        "IR_097": 264.493309404708,
        "IR_108": 284.52530213584566,
        "IR_120": 282.8899476201401,
        "IR_134": 262.26011396403493,
        "WV_062": 240.32356644848787,
        "WV_073": 257.80915140156566,
        "satellite_zenith": 29.680696852618667
    },
    "standard_deviation": {
        "IR_087": 17.892575448132664,
        "IR_097": 12.28670943291758,
        "IR_108": 19.583220627917424,
        "IR_120": 19.756096100575007,
        "IR_134": 13.160031361493516,
        "WV_062": 8.421246366287647,
        "WV_073": 11.312417909693224,
        "satellite_zenith": 13.966242106708128
    }
}

FILL_VALUE = -999_999

LONLAT_RES = 0.027 # Assumed spatial resolution in degrees

SEVIRI_0DEG_AREADEF = get_area_def('msg_seviri_fes_3km')
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="^You will likely lose important projection information when converting to a PROJ string from another format"
    )
    SEVIRI_0DEG_PROJ4 = SEVIRI_0DEG_AREADEF.proj_dict

# Data until December 2017 can suffer an offset
# https://github.com/pytroll/satpy/blob/d3e6fd4ae3d0a30c8dc46ba28546a9eb2ae55b3e/satpy/readers/seviri_l1b_native.py#L460-L469
SEVIRI_0DEG_AREADEF_OFFSET = AreaDefinition(
    area_id=SEVIRI_0DEG_AREADEF.area_id,
    description=SEVIRI_0DEG_AREADEF.description,
    proj_id=SEVIRI_0DEG_AREADEF.proj_id,
    projection=SEVIRI_0DEG_AREADEF.crs,
    width=SEVIRI_0DEG_AREADEF.width,
    height=SEVIRI_0DEG_AREADEF.height,
    area_extent=np.array(SEVIRI_0DEG_AREADEF.area_extent) + 1500 * np.array([1, -1, 1, -1]),
    dtype=SEVIRI_0DEG_AREADEF.dtype,
)

REGION_OF_INTEREST = {
    'lat_min': -40,
    'lat_max': 40,
    'lon_min': -20,
    'lon_max': 55,
    # Idxs determined with SEVIRI_0DEG_AREADEF
    'x_min_idx': 1138,
    'x_max_idx': 3464,
    'y_min_idx': 557,
    'y_max_idx': 3155
    # If determined with area given when reading a file with satpy,
    # the area is flipped, then the idxs are:
    # 'x_min_idx': 247,
    # 'x_max_idx': 2573,
    # 'y_min_idx': 556,
    # 'y_max_idx': 3154
}

# Note that the AreaDefinition below was manually determined,
# which uses that IMERG has a resolution of 0.1 degrees
# at degrees [..., -0.15, -0.05, 0.05, 0.15, ...]
# and the ROI is defined by integers
IMERG_AREADEF_ROI = AreaDefinition(
    area_id='Africa',
    description='IMERGE_ROI',
    proj_id=None,
    projection='epsg:4326',
    width=750,
    height=800,
    area_extent=[
        REGION_OF_INTEREST['lon_min'], REGION_OF_INTEREST['lat_min'],
        REGION_OF_INTEREST['lon_max'], REGION_OF_INTEREST['lat_max']
    ]
)

# The dictionary below, when used to index a satpy.Scene
# of a SEVIRI 0 degree image, i.e. using
# xarray.DataAarray[AREA_EXTENT_INDXS_SEVIRI_2048]
# will return a 2048x2048 pixels image that covers
# most of Africa with little distortion
AREA_EXTENT_INDXS_SEVIRI_2048 = {
    'y': np.arange(-900 - 2048, -900),
    'x': np.arange(-1200 - 2048, -1200)
}

SEVIRI_0DEG_AREADEF_2048 = AreaDefinition(
        area_id=None, description=None,
        proj_id=None, projection=SEVIRI_0DEG_PROJ4,
        width=2048, height=2048,
        # Numbers calculated manually using
        # AREA_EXTENT_INDXS_SEVIRI_2048
        # boundary box is 'reversed' because
        # data provided with Satpy is flipped
        area_extent=[
            4175061.00523472,
            2869885.62810421,
            -1969764.67835903,
            -3274940.05548954
        ]
    )

INFERENCE_KEYS = [CHANNELS_MAP[k] for k in sorted(CHANNELS_MAP.keys())] + ["satellite_zenith"]

TRANSFORMS_INPUT = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[TRAINING_SET_STATISTICS["mean"][k] for k in INFERENCE_KEYS],
            std=[TRAINING_SET_STATISTICS["standard_deviation"][k] for k in INFERENCE_KEYS]
        )
    ]
)

def prepare_dataset_for_network(ds: xr.Dataset, fill_value: float=FILL_VALUE) -> torch.tensor:
    """
    Prepare a dataset for the network by stacking the channels, normalizing them,
    and filling NaNs with a fill value.

    Args:
        ds: dataset containing the variables to prepare
        fill_value: value to use to fill NaNs
    
    Returns:
        A torch tensor with the variables stacked, normalized, and NaNs filled.
    """
    x = np.stack([ds[v].values for v in INFERENCE_KEYS], axis=-1)
    x = np.where(x <= -999_999, np.nan, x)
    x = TRANSFORMS_INPUT(x)
    x[torch.isnan(x)] = fill_value
    return x.float()

    
def to_latlon(ds: xr.Dataset, lonlat_res: float=LONLAT_RES, area_def_eqc: AreaDefinition=None) -> xr.Dataset:
    """
    Map data on the SEVIRI grid to a regular lat-lon grid.

    Args:
        ds: dataset containing data in SEVIRI grid
        lonlat_res: the resolution to use for the regular grid
    
    Returns:
        The dataset with the variables updated to match the regular lat-lon grid.
    """

    transformer = Transformer.from_crs(
        SEVIRI_0DEG_PROJ4,
        "epsg:4326",
        always_xy=True
    )

    # area_extent is flipped in SEVIRI projection
    lon_min, lat_min = transformer.transform(
        ds.area_extent[-2], ds.area_extent[-1]
    )

    x_size = ds.x.size
    y_size = ds.y.size

    if area_def_eqc is None:
        area_extent = [
            lon_min,
            lat_min,
            lon_min + lonlat_res * x_size,
            lat_min + lonlat_res * y_size
        ]
        area_def_eqc = AreaDefinition(
                area_id='eqc', description = 'Equidistant Cylindrical',
                proj_id = 'eqc', projection = 'epsg:4326', 
                width=x_size, height=y_size,
                area_extent=area_extent
        )

    area_def_seviri = AreaDefinition(
        area_id='seviri_0deg', description='seviri_0deg',
        proj_id='seviri_0deg', projection=SEVIRI_0DEG_PROJ4,
        width=x_size, height=y_size,
        area_extent=ds.area_extent
    )

    resampled_data = resample_nearest(
        area_def_seviri,
        np.stack([ds[v] for v in ds.keys()], axis=-1),
        area_def_eqc,
        radius_of_influence=6e3,
        fill_value=np.nan
    )

    return xr.Dataset(
        data_vars={k: (("y", "x"), resampled_data[..., i]) for i, k in enumerate(ds.keys())},
        coords={
            "x": area_def_eqc.projection_x_coords,
            "y": area_def_eqc.projection_y_coords
        }
    ).expand_dims("nat_file").assign_coords(nat_file=[ds.nat_file.values])

@jit(nopython=True)
def satellite_angles(lons_o: np.ndarray[np.float64], lats_o: np.ndarray[np.float64],
                     lons_s: np.ndarray[np.float64], lats_s: np.ndarray[np.float64],
                     h_s: np.ndarray[np.float64], a: float, b: float) -> \
                         tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """Compute the satellite zenith and azimuth view angles (north-clockwise).
    The observer is assumed to be at 0 geodetic height.
    
    Args:
        lons_o: geodetic longitudes of the observer, in degrees
        lats_o: geodetic latitudes of the observer, in degrees
        lons_s: geodetic longitudes of the satellite, in degrees
        lats_s: geodetic latitudes of the satellite, in degrees
        h_s: geodetic heights of the satellite, in metres
        a: equatorial radius (Earth's semimajor axis), in metres
        b: polar radius (Earth's semiminor axis), in metres
    
    Returns a tuple with the (azimuth, zenith) satellite angles in degrees, where
        each angle is a numpy ndarray of the same shape as the inputs.
    """

    # Degrees to radians
    # Could also have use np.deg2rad and np.rad2deg
    D2R = np.pi / 180

    # Curviliniear geodetic coordinates in degrees
    get_x = lambda N, h, lat, lon: (N + h) * np.cos(lat * D2R) * np.cos(lon * D2R)
    get_y = lambda N, h, lat, lon: (N + h) * np.cos(lat * D2R) * np.sin(lon * D2R)
    get_z = lambda N, h, lat, e2: ((N * (1 - e2) + h) * np.sin(lat * D2R))

    # (Squared) eccentricity
    e2 = 1 - (b / a) ** 2
    
    # Compute curvilinear geodetic coordinates for observer
    N_o = a / np.sqrt(1 - e2 * (np.sin(lats_o * D2R) ** 2))
    h_o = 0
    x_o = get_x(N_o, h_o, lats_o, lons_o)
    y_o = get_y(N_o, h_o, lats_o, lons_o)
    z_o = get_z(N_o, h_o, lats_o, e2)

    # Compute curvilinear geodetic coordinates for satellite
    N_s = a / np.sqrt(1 - e2 * (np.sin(lats_s * D2R) ** 2))
    x_s = get_x(N_s, h_s, lats_s, lats_s)
    y_s = get_y(N_s, h_s, lats_s, lons_s)
    z_s = get_z(N_s, h_s, lats_s, e2)

    # Vector from the observer to the satellite in ECEF system
    # Earth Centre Earth Fixed
    x = x_s - x_o
    y = y_s - y_o
    z = z_s - z_o

    # Vector from the observer to the satellite in ENU system
    # East, North, Upwards
    # See https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU
    # The matrix can be derived by first a rotation of '-longitude' around the Z
    # axis and then a rotation of 'latitude' around the rotated Y axis
    e =                        - np.sin(lons_o * D2R) * x + np.cos(lons_o * D2R) * y
    n = - np.sin(lats_o * D2R) * np.cos(lons_o * D2R) * x - np.sin(lats_o * D2R) * np.sin(lons_o * D2R) * y + np.cos(lats_o * D2R) * z
    u =   np.cos(lats_o * D2R) * np.cos(lons_o * D2R) * x + np.sin(lons_o * D2R) * np.cos(lats_o * D2R) * y + np.sin(lats_o * D2R) * z

    # Azimuth, preserving angle quadrant, north-clockwise convention
    azimuth = np.arctan2(e, n) / D2R

    # Zenith
    zenith = np.arccos(u / np.sqrt(e**2 + n**2 + u**2)) / D2R

    return (azimuth, zenith)

class MSGNative:
    """
    Class to load MSG data in native format using Satpy, selecting only the channels
    used in the RoA paper, and adding the satellite zenith angle.

    Args:
        file: path to the MSG file to load
    """
    def __init__(self, file: str):
        self.file = file
        self.reader = 'seviri_l1b_native'


    def get_dataset(self,
                    area_extent: dict[str, int]={
                        'x': np.arange(261 , 2565),
                        'y': np.arange(511, 3199)
                    }
        ) -> xr.Dataset:
        """
        Load the dataset and return the channels and satellite zenith angle.

        Args:
            area_extent: area extent to use, default is the region of interest.
        
        Returns:
            Dataset with the channels and satellite zenith angle.
        """
        filenames = [self.file] if self.reader == 'seviri_l1b_native' else self.file
        scn = Scene(filenames=filenames, reader=self.reader)
        scn.load(CHANNELS_MAP.values())
        ds = xr.merge(
            (
                xr.merge(
                    [scn[name].reset_coords(drop=True).to_dataset(name=name) for name in CHANNELS_MAP.values()]
                ),
                # Use an arbitrary channel to obtain the line acquisition time
                scn['WV_062'].acq_time.reset_coords(names='acq_time').reset_coords(drop=True)
            )
        )
        if area_extent:
            ds = ds[{'y': area_extent['y'], 'x': area_extent['x']}]
        x, y = np.broadcast_arrays(ds.x.data.reshape(1, -1), ds.y.data.reshape(-1, 1))
        # Use arbitrary channel to obtain orbital parameters
        lons_o, lats_o = SEVIRI_0DEG_AREADEF.get_lonlat_from_projection_coordinates(x, y)
        orbital_parameters = scn['WV_062'].attrs['orbital_parameters']
        satellite_zenith = satellite_angles(
            lons_o,
            lats_o,
            orbital_parameters['satellite_actual_longitude'],
            orbital_parameters['satellite_actual_latitude'],
            orbital_parameters['satellite_actual_altitude'],
            SEVIRI_0DEG_PROJ4['a'],
            SEVIRI_0DEG_PROJ4['a'] * (1 - 1 / SEVIRI_0DEG_PROJ4['rf'])
        )[1].astype(np.float32)
        return xr.merge(
            (
                ds,
                xr.Dataset(
                    data_vars={
                        'satellite_zenith': (("y", "x"), satellite_zenith)
                    },
                    coords={
                        "x": ds.x.data,
                        "y": ds.y.data
                    }
                )
            )
        )
    

class MSGHRIT(MSGNative):
    """
    Class to load MSG data in HRIT format using Satpy, selecting only the channels
    used in the RoA paper, and adding the satellite zenith angle.

    Args:
        file: path to the MSG observation in HRIT format
    
    Notes:
        The `file` argument is a list of files when using MSGHRIT. See
        https://user.eumetsat.int/catalogue/EO:EUM:DAT:MSG:HRSEVIRI/access
        and
        https://satpy.readthedocs.io/en/stable/api/satpy.readers.seviri_l1b_hrit.html
    """
    def __init__(self, file: list[str]):
        self.file = file
        self.reader = 'seviri_l1b_hrit'
    

class MSGDataset(Dataset):
    """PyTorch Dataset to load MSG data for evaluation or inference.

    Args:
        path: path to directory of netCDF files to glob and use.
        gpm_precip: return reference data.
        area_extent: Area extent to use as
            (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
        fill_value: input NaNs will be masked with this value.
        resample: use the regular lat-lon grid.
        train: return data with data augmentation for training

    Returns:
        Standardized inputs for the network, with channels sorted as
            [MSG channels] + [satellite_zenith]. If specified, also the reference
            precipitation. The data is sorted based on the filenames.
    """

    def __init__(self, path: str, gpm_precip: bool=False,
                 area_extent: list[float]=None, fill_value: float=FILL_VALUE,
                 resample: bool=False, train: bool=False):
        self.files = sorted(list(Path(path).rglob('*nc')))
        self.gpm_precip = gpm_precip
        self.fill_value = fill_value
        self.resample = resample

        self.transforms_input = TRANSFORMS_INPUT
        self.transforms_reference = transforms.Compose([transforms.ToTensor()])
        if area_extent:
            raise NotImplementedError
        
        self.train = train
        if self.train:
            self.gpm_precip = True
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with xr.open_dataset(self.files[idx]) as ds:
            if self.resample:
                x = prepare_dataset_for_network(to_latlon(ds), fill_value=self.fill_value)
            else:
                x = prepare_dataset_for_network(ds, fill_value=self.fill_value)
        
        if self.gpm_precip:
            y = self.transforms_reference(ds["gpm_precip"].values)

            if self.train:
                x, y = self._data_augmentation(x, y)
            
            return x, y
        
        return x
    
    def _data_augmentation(self, x, y):
        # Random number generator
        rng = np.random.default_rng()
        
        # Pick rotation
        deg = rng.choice([0, 90, 180, 270])
        rotation = transforms.RandomRotation(degrees=(deg, deg))
        
        # Pick probabilities to flip
        p_flip_h = rng.choice([0, 1])
        flip_h = transforms.RandomHorizontalFlip(p=p_flip_h)
        p_flip_v = rng.choice([0, 1])
        flip_v = transforms.RandomVerticalFlip(p=p_flip_v)

        # Random crop transforms
        i,j,h,w = transforms.RandomCrop.get_params(x, output_size=(128,128))

        # Apply transforms
        da_transforms = transforms.Compose([rotation, flip_h, flip_v])

        x = transforms.functional.crop(da_transforms(x), i, j, h, w)
        y = transforms.functional.crop(da_transforms(y), i, j, h, w)

        return x, y
    

class GPM:
    """
    Class to load GPM data using pansat.

    Args:
        file_path: path to the GPM file to load
        resampled: resample the data to the regular lat-lon grid
    """
    def __init__(self, file_path: Path, resampled: bool=False):
        self.dataset = l2b_gpm_cmb.open(file_path)
        self.resampled = resampled
        if self.resampled:
            self.resample()

    def resample(self):
        scan_time_broadcasted = np.broadcast_to(
            self.dataset.scan_time.values.reshape(-1, 1),
            (self.dataset.scan_time.values.size, self.dataset.matched_pixels.size)
        )
        data = np.stack([self.dataset.near_surf_precip_tot_rate.values, scan_time_broadcasted.astype(int)], axis=-1)
        resampled_data = resample_nearest(
            SwathDefinition(self.dataset.longitude.data, self.dataset.latitude.data),
            data,
            SEVIRI_0DEG_AREADEF,
            radius_of_influence=3_600,
            fill_value=np.nan
        )
        dims = ("y", "x")
        self.dataset_resampled = xr.Dataset(
            data_vars={
                "near_surf_precip_tot_rate": (dims, resampled_data[..., 0]),
                "scan_time": (dims, resampled_data[..., 1].astype('datetime64[ns]')),
            },
            coords={
                "x": SEVIRI_0DEG_AREADEF.projection_x_coords,
                "y": SEVIRI_0DEG_AREADEF.projection_y_coords
            }
        )
        self.resampled = True


class GPM2BCMB:
    """
   Class to load GPM 2BCMB data without pansat.

    Args:
        file_path: path to the GPM file to load. Also works with earthaccess granules.
    """
    def __init__(self, file_path: Path | str) -> None:
        self.file_path = Path(file_path)
    
    def read(self, variables: list=['nearSurfPrecipTotRate']) -> xr.Dataset:
        variables = variables + ['Latitude', 'Longitude']
        ds = xr.open_dataset(self.file_path, group='KuGMI')[set(variables)]
        ds = ds.rename_dims({k_old: k_new for k_old, k_new in zip(sorted(ds.sizes, key=ds.sizes.get), ['matched', 'points'])})
        return ds.assign_coords(matched=np.arange(49), points=self.get_timestamps().timestamp.values)

    def get_timestamps(self) -> xr.Dataset:
        ds = xr.open_dataset(self.file_path, group='KuGMI/ScanTime')
        digits = zip(
            ds.Year.values.astype(int),
            ds.Month.values.astype(int),
            ds.DayOfMonth.values.astype('timedelta64[D]').astype(int),
            ds.Hour.values.astype('timedelta64[h]').astype(int),
            ds.Minute.values.astype('timedelta64[m]').astype(int),
            ds.Second.values.astype(int),
            ds.MilliSecond.values.astype(int)
        )
        dates = []
        for d in digits:
            d = list(d)
            # The if conditions handle spurious cases
            # There's at least one case that yields 2015-06-30T23:59:60.200 otherwise
            dt = np.timedelta64(0)
            if d[5] >= 60:
                d[5] = 0
                dt += np.timedelta64(1, 'm')
            if d[4] >= 60:
                d[4] = 0
                dt += np.timedelta64(1, 'h')
            if d[3] >= 24:
                d[3] = 0
                dt += np.timedelta64(1, 'D')
            dates.append(
                np.datetime64(f"{d[0]}-{d[1]:02d}-{d[2]:02d}T{d[3]:02d}:{d[4]:02d}:{d[5]:02d}.{d[6]:03d}") + dt
            )
        return xr.DataArray(np.array(dates).astype('datetime64[ns]'), dims=ds.dims, name='timestamp').to_dataset()