from cosipy.util.iterables import asarray
import numpy as np

def test_asarray():

    # From array
    a = np.asarray([1,2,3], dtype = int)

    # Same dtype. No copy
    b = asarray(a, int)
    assert isinstance(b, np.ndarray)
    assert b.dtype == int
    assert np.shares_memory(a, b)
    assert np.allclose(b, a)

    # Different dtype
    b = asarray(a, np.float64)
    assert isinstance(b, np.ndarray)
    assert b.dtype == np.float64
    assert np.allclose(b, a)

    # Soft dtype requirement. No copy
    b = asarray(a, np.float64, force_dtype=False)
    assert isinstance(b, np.ndarray)
    assert b.dtype == int
    assert np.shares_memory(a,b)
    assert np.allclose(b, a)

    # From array-like
    a = [1,2,3]

    # With force dtype
    b = asarray(a, np.float64)
    assert isinstance(b, np.ndarray)
    assert b.dtype == np.float64
    assert np.allclose(b, a)

    b = asarray(a, np.int32)
    assert isinstance(b, np.ndarray)
    assert b.dtype == np.int32
    assert np.allclose(b, a)

    # Relax dtype. No copy
    b = asarray(a, np.float64, force_dtype=False)
    assert isinstance(b, np.ndarray)
    assert b.dtype == np.asarray(a).dtype # numpy infers int64
    assert np.allclose(b, a)

    # From generator w/o len
    a_list = [1,2,3]
    def gen():
        for i in a_list:
            yield i

    a = gen()
    b = asarray(a, np.float64)
    assert isinstance(b, np.ndarray)
    assert b.dtype == np.float64
    assert np.allclose(b, a_list)

    a = gen()
    b = asarray(a, np.int32)
    assert isinstance(b, np.ndarray)
    assert b.dtype == np.int32
    assert np.allclose(b, a_list)
