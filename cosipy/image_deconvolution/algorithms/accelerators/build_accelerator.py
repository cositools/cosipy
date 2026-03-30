"""
Factory that maps algorithm names to AcceleratorBase subclasses.

Adding a new accelerator
------------------------
1. Implement a subclass of AcceleratorBase.
2. Import it here.
3. Add the name -> class mapping to accelerator_classes.
"""

import logging

from .accelerator_base import AcceleratorBase
from .max_step_accelerator import MaxStepAccelerator
from .line_search_accelerator import LineSearchAccelerator

logger = logging.getLogger(__name__)

ACCELERATOR_CLASSES : dict = {
    "MaxStep"   : MaxStepAccelerator,
    "LineSearch": LineSearchAccelerator,
}

DEFAULT_ACCELERATOR = "MaxStep"


def build_accelerator(parameter) -> AcceleratorBase | None:
    """
    Construct an AcceleratorBase from the ``acceleration`` block.

    Parameters
    ----------
    parameter : dict-like
        activate: true
        algorithm: MaxStep # optional, default MaxStep
        accel_factor_max: 10.0

    Returns
    -------
    AcceleratorBase

    Raises
    ------
    ValueError
        When the requested algorithm name is not in ACCELERATOR_CLASSES.
    """

    algorithm_name = parameter.get("algorithm", DEFAULT_ACCELERATOR)

    if algorithm_name not in ACCELERATOR_CLASSES:
        available = ", ".join(ACCELERATOR_CLASSES.keys())
        raise ValueError(
            f'Unknown accelerator "{algorithm_name}". Available: {available}'
        )

    cls         = ACCELERATOR_CLASSES[algorithm_name]
    accelerator = cls(parameter)

    logger.info(f"[Accelerator '{algorithm_name}' created]")
    return accelerator

