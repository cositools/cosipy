import pytest
import cosipy

if not cosipy.with_ml:
    pytest.skip(reason="Optional [ml] dependencies not installed", allow_module_level=True) 

from cosipy.threeml.ml.function_torch import FastPowerlawPyTorch, FastGaussianPyTorch
import astropy.units as u
import numpy as np
import torch 

# ==========================================
# Tests for FastPowerlawPyTorch
# ==========================================

def test_powerlaw_basic_numeric():
    """Test Powerlaw evaluation with standard floats and numpy arrays (no units)."""
    model = FastPowerlawPyTorch()
    
    x = np.array([1.0, 2.0, 4.0])
    K = 2.0
    piv = 1.0
    index = -2.0
    
    # Expected: 2.0 * (x / 1.0)^(-2.0) -> [2.0, 0.5, 0.125]
    expected = K * (x / piv) ** index
    result = model.evaluate(x, K, piv, index)
    
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result.flatten(), expected, rtol=1e-6)


def test_powerlaw_with_astropy_units():
    """Test Powerlaw when parameters and inputs are passed as Astropy Quantities."""
    model = FastPowerlawPyTorch()
    
    # Manually mimic what astromodels sets up behind the scenes for units
    model._y_unit = u.erg / (u.cm**2 * u.s * u.keV)
    model._x_unit = u.keV
    
    x = np.array([10.0, 20.0]) * u.keV
    K = 3.0 * model._y_unit
    piv = 10.0 * u.keV
    index = -2.0 * u.dimensionless_unscaled
    
    # Expected: 3.0 * (x / 10.0)^(-2.0)
    expected_values = 3.0 * (x.value / 10.0) ** -2.0
    
    result = model.evaluate(x, K, piv, index)
    
    assert isinstance(result, u.Quantity)
    np.testing.assert_allclose(result.flatten().value, expected_values, rtol=1e-6)
    assert result.unit == model._y_unit


# ==========================================
# Tests for FastGaussianPyTorch
# ==========================================

def test_gaussian_docstring_case_1():
    """Verifies the first test case defined in the docstring: x=0.0"""
    model = FastGaussianPyTorch()
    
    # Standard normal distribution parameters
    F = 1.0
    mu = 0.0
    sigma = 1.0
    
    result = model.evaluate(0.0, F, mu, sigma)
    
    # Check against docstring value: 0.3989422804014327, tolerance: 1e-10
    assert abs(result - 0.3989422804014327) < 1e-10


def test_gaussian_docstring_case_2():
    """Verifies the second test case defined in the docstring: x=-1.0"""
    model = FastGaussianPyTorch()
    
    F = 1.0
    mu = 0.0
    sigma = 1.0
    
    result = model.evaluate(-1.0, F, mu, sigma)
    
    # Check against docstring value: 0.24197072451914337, tolerance: 1e-9
    assert abs(result - 0.24197072451914337) < 1e-9


def test_gaussian_array_input():
    """Ensure Gaussian can evaluate arrays and returns correct shape."""
    model = FastGaussianPyTorch()
    
    x = np.array([-1.0, 0.0, 1.0])
    F = 2.5
    mu = 0.5
    sigma = 1.5
    
    result = model.evaluate(x, F, mu, sigma)
    
    assert result.shape == (3,)
    
    # Calculate pure numpy alternative to verify values
    norm = (1.0 / np.sqrt(2 * np.pi)) / sigma
    expected = F * norm * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    
    np.testing.assert_allclose(result.flatten(), expected, rtol=1e-6)
    
