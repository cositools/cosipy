import cosipy

if not cosipy.with_ml:
    raise ImportError("Install cosipy with [ml] optional packages to use these features.")

from .ContinuumEstimationNN import ContinuumEstimationNN
from .ContinuumEstimationNN import GCN