import contextlib

import numpy as np
from scipy.stats import loguniform
import torch

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