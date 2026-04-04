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
from .background_estimation import TransientBackgroundEstimation

# Flag to detect whether cosipy has all optional machine learning [ml] dependencies
# If you update the packages here, make sure to do the same changes in
# pyproject.toml project.optional-dependencies.ML
def _with_ml():
    from importlib.util import find_spec
    ml_pkg = [
        "torch",
        "torch_geometric"
    ]
    with_ml = True
    for pkg in ml_pkg:
        if not find_spec(pkg):
            with_ml = False
            break

    return with_ml

with_ml = _with_ml()