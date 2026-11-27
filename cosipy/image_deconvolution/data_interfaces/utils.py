"""
Utility functions for data interfaces.
"""

# Moved to cosipy.util.tensor, since it's a generic numeric helper
# used outside of image_deconvolution too (e.g. segmented extended-
# source spectral fitting). Re-exported here for backward compatibility.
from cosipy.util.tensor import tensordot_sparse
