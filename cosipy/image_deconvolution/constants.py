"""
Constants for the image deconvolution module.
This module centralizes magic numbers and default values.
"""
import astropy.units as u

# Numerical Constants
NUMERICAL_ZERO = 1e-12 # Small value to avoid division by zero in expectation calculations.
CHUNK_SIZE_FITS = 998 # Maximum columns in FITS table (FITS limit is 1000, using 998 for safety).

# Physical Constants
EARTH_RADIUS_KM = 6378.0 # Earth radius in km

# Default Values - General Deconvolution Algorithm Parameters
DEFAULT_MINIMUM_FLUX = 0.0 # Default minimum flux to enforce non-negativity.
DEFAULT_ITERATION_MAX = 1 # Default maximum number of iterations.
DEFAULT_STOPPING_THRESHOLD = 1e-2 # Default convergence threshold for log-likelihood change.

# Default Values - Background Normalization
DEFAULT_BKG_NORM_RANGE = [0.0, 100.0] # Default allowed range [min, max] for background normalization factors.

# Default Values - Response Weighting
DEFAULT_RESPONSE_WEIGHTING_INDEX = 0.5 # Default power index for response weighting: filter = (exposure/max)^index.
