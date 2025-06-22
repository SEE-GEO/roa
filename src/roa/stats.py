from ipwgml.metrics import iterate_windows
import numpy as np
from scipy.fft import fftn, fftfreq, fftshift
from scipy.stats import rankdata
import torch
import xarray as xr

from roa.utils import temp_seed

def spearman_correlation_2d(x: np.ndarray, y: np.ndarray, axis: int=1) -> np.ndarray:
    """
    Compute the Spearman correlation coefficient between two 2-dimensional NumPy arrays
    of shape (n_obs, n_vars) or (n_vars, n_obs)

    Parameters:
        x, y: input arrays with the same shape
        axis: axis where n_vars are stored
    
    Returns:
        rho: Spearman correlation coefficient tensor
    """
    
    x_rank = rankdata(x, axis=axis).astype(x.dtype)
    y_rank = rankdata(y, axis=axis).astype(y.dtype)

    cov = (
        (
            x_rank - x_rank.mean(axis=axis, keepdims=True)
        )*(
            y_rank - y_rank.mean(axis=axis, keepdims=True)
        )
    ).mean(axis=axis)

    denominator = x_rank.std(axis=axis) * y_rank.std(axis=axis)

    return np.divide(
        cov,
        denominator,
        out=np.full_like(cov, np.nan),
        where=np.isfinite(denominator) * (~np.isclose(denominator, 0))
    )


def spearman_correlation_4d(x, y):
    """
    Compute the Spearman correlation coefficient between two 4-dimensional PyTorch tensors.
    Assumes axis to be sorted is 1
    
    Parameters:
        x, y: input tensors with the same shape

    Returns:
        rho: Spearman correlation coefficient tensor.
    """
    # Ensure the tensors are floating point type
    x = x.float()
    y = y.float()

    # Check if any is invalid (constant)
    undefined_x = torch.where(
        torch.all(torch.eq(x, x[:, [0], ...]), dim=1),
        True,
        False
    )
    undefined_y = torch.where(
        torch.all(torch.eq(y, y[:, [0], ...]), dim=1),
        True,
        False
    )
    undefined = torch.logical_or(undefined_x, undefined_y)

    # Compute ranks
    x_rank = x.argsort(dim=1).argsort(dim=1).float()
    y_rank = y.argsort(dim=1).argsort(dim=1).float()

    # Compute mean ranks
    x_mean_rank = x_rank.mean(dim=1, keepdim=True)
    y_mean_rank = y_rank.mean(dim=1, keepdim=True)

    # Compute covariance and standard deviations
    cov = ((x_rank - x_mean_rank) * (y_rank - y_mean_rank)).mean(dim=1, keepdim=True)
    x_std = torch.sqrt(((x_rank - x_mean_rank) ** 2).mean(dim=1, keepdim=True))
    y_std = torch.sqrt(((y_rank - y_mean_rank) ** 2).mean(dim=1, keepdim=True))

    # Compute Spearman correlation coefficient
    rho = cov / (x_std * y_std)

    # Apply undefined values
    rho = torch.where(undefined, torch.tensor(float('nan')), rho)

    return rho

class FourierSpectralDensity:
    """
    Based on ipwgml.metrics.SpectralCoherence
    """

    def __init__(self, window_size: int, scale: float):
        """
        Args:
            window_size: The size of the window over which the coefficients are computed.
            scale: Spatial extent of a single pixel.
        """
        self.window_size = window_size
        self.freq_x = fftshift(fftfreq(window_size, scale))
        self.freq_y = fftshift(fftfreq(window_size, scale))
        self.coeffs_target_sum = np.zeros((window_size, window_size), dtype=np.complex128)
        self.coeffs_target_sum2 = np.zeros((window_size, window_size), dtype=np.float64)
        self.coeffs_pred_sum = np.zeros((window_size, window_size), dtype=np.complex128)
        self.coeffs_pred_sum2 = np.zeros((window_size, window_size), dtype=np.float64)
        self.coeffs_targetpred_sum = np.zeros((window_size, window_size), dtype=np.complex128)
        self.coeffs_targetpred_sum2 = np.zeros((window_size, window_size), dtype=np.float64)
        self.coeffs_diffs_sum = np.zeros((window_size, window_size), dtype=np.complex128)
        self.coeffs_diffs_sum2 = np.zeros((window_size, window_size), dtype=np.float64)
        self.counts = np.zeros((window_size, window_size), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray, seed: int=None):
        """
        Calculate spectral statistics for all valid sample windows in
        given results.

        Args:
            pred: A np.ndarray containing the predicted precipitation field.
            target: A np.ndarray containing the reference data.
            seed: Seed for reproducibility.
        """
        valid = np.isfinite(target)
        with temp_seed(seed):
            # iterate_windows uses np.random.choice
            # allow for setting seed for reproducibility
            for rect in iterate_windows(valid, self.window_size):
                row_start, col_start, row_end, col_end = rect
                pred_w = pred[row_start:row_end, col_start:col_end]
                target_w = target[row_start:row_end, col_start:col_end]
                w_pred = fftshift(fftn(pred_w, norm="ortho"))
                w_target = fftshift(fftn(target_w, norm="ortho"))
                self.coeffs_target_sum += w_target
                self.coeffs_target_sum2 += np.abs(w_target * w_target.conj())
                self.coeffs_pred_sum += w_pred
                self.coeffs_pred_sum2 += np.abs(w_pred * w_pred.conj())
                self.coeffs_targetpred_sum += w_target * w_pred.conj()
                self.coeffs_targetpred_sum2 += np.abs(w_target * w_pred.conj() * (w_target * w_pred.conj()).conj())
                self.coeffs_diffs_sum += w_target - w_pred
                self.coeffs_diffs_sum2 += np.abs(self.coeffs_diffs_sum * self.coeffs_diffs_sum.conj())
                self.counts += np.isfinite(w_pred)

    def to_dataset(self):
        """
        Return the data as an xarray.Dataset.
        """
        return xr.Dataset(
            data_vars={
                'coeffs_target_sum': (('freqs_y', 'freqs_x'), self.coeffs_target_sum),
                'coeffs_target_sum2': (('freqs_y', 'freqs_x'), self.coeffs_target_sum2),
                'coeffs_pred_sum': (('freqs_y', 'freqs_x'), self.coeffs_pred_sum),
                'coeffs_pred_sum2': (('freqs_y', 'freqs_x'), self.coeffs_pred_sum2),
                'coeffs_targetpred_sum': (('freqs_y', 'freqs_x'), self.coeffs_targetpred_sum),
                'coeffs_targetpred_sum2': (('freqs_y', 'freqs_x'), self.coeffs_targetpred_sum2),
                'coeffs_diffs_sum': (('freqs_y', 'freqs_x'), self.coeffs_diffs_sum),
                'coeffs_diffs_sum2': (('freqs_y', 'freqs_x'), self.coeffs_diffs_sum2),
                'counts': (('freqs_y', 'freqs_x'), self.counts),
            },
            coords={
                'freqs_x': self.freq_x,
                'freqs_y': self.freq_y
            }
        )