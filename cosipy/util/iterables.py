import itertools
from typing import Union, Iterable, Optional
import numpy.typing as npt

import numpy as np


def itertools_batched(iterable, n, *, strict=False):
    """
    itertools.batched was added in version 3.12.
    Use the "roughly equivalent" from itertools documentation for now.
    """

    # batched('ABCDEFG', 2) → AB CD EF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

def asarray(a : Union[npt.ArrayLike, Iterable], dtype:npt.DTypeLike, force_dtype = True):
    """
    Convert an iterable or an array-like object into a numpy array.

    Parameters
    ----------
    a: Iterable or array-like object
    dtype: Desired type (e.g. np.float64)
    force_dtype: If True, it is guaranteed that the output will have the specified dtype. If False,
        we will attempt to infer the data-type from the input data, and dtype will be considered a fallback option.
        Relaxing the dtype requirement can prevent an unnecessary copy if the input type does not exactly match the
        requested dtype (e.g. np.float32 vs np.float64)

    Returns
    -------

    """
    if hasattr(a, "__len__"):
        # np.asarray does not work with an object without __len__
        if not force_dtype:
            # the data-type is inferred from the input data.
            dtype = None

        return np.asarray(a, dtype = dtype)
    else:
        # fromiter needs a dtype
        return np.fromiter(a, dtype = dtype)