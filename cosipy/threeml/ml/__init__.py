import cosipy

if not cosipy.with_ml:
    raise ImportError("Install cosipy with [ml] optional packages to use these features, e.g., \"pip install cosipy[ml]\".")

from .optimized_unbinned_folding import UnbinnedThreeMLPointSourceResponseIRFAdaptive, UnbinnedThreeMLPointSourceResponseIRFAdaptiveBuckets