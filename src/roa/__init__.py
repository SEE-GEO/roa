from importlib.metadata import version, PackageNotFoundError
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

try:
    __version__ = version("roa")
except PackageNotFoundError:
    __version__ = "unknown"

from .data import (
    MSGNative,
    QUANTILES,
    prepare_dataset_for_network
)
from .tiler import Tiler