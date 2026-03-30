try:
    from importlib import metadata
    __version__ = metadata.version("cosipy")
except metadata.PackageNotFoundError:
    # Handle cases where the package is not installed (e.g., running directly from source)
    __version__ = "unknown"

from .response import DetectorResponse

from .spacecraftfile import *

from .data_io import DataIO
from .data_io import UnBinnedData
from .data_io import BinnedData
from .data_io import ReadTraTest

from .threeml import Band_Eflux

from .spacecraftfile import SpacecraftHistory

from .ts_map import FastTSMap, MOCTSMap

from .source_injector import SourceInjector

from .background_estimation import LineBackgroundEstimation
from .background_estimation import ContinuumEstimation
from .background_estimation import TransientBackgroundEstimation

