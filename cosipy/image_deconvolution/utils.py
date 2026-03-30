"""
Shared utility functions for the image_deconvolution module.
"""

def _to_float(x) -> float:
    """
    Convert to float, handling astropy Quantity.
    """
    if hasattr(x, 'unit'):
        return float(x.to(''))
    return float(x)
