import pytest
import cosipy

if not cosipy.with_ml:
    pytest.skip(reason="Optional [ml] dependencies not installed", allow_module_level=True) 

from cosipy.threeml.ml.function_torch import FastPowerlawPyTorch, FastGaussianPyTorch, FastCutoffPowerlawPyTorch, FastSuperCutoffPowerlawPyTorch
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
    
    model.set_units(
        u.keV,
        1 / (u.keV * u.cm ** 2 * u.s),
    )
    
    x = np.array([10.0, 20.0]) * u.keV
    K = 3.0 * model.y_unit
    piv = 10.0 * u.keV
    index = -2.0 * u.dimensionless_unscaled
    
    # Expected: 3.0 * (x / 10.0)^(-2.0)
    expected_values = 3.0 * (x.value / 10.0) ** -2.0
    
    result = model.evaluate(x, K, piv, index)
    
    assert isinstance(result, u.Quantity)
    np.testing.assert_allclose(result.flatten().value, expected_values, rtol=1e-6)
    assert result.unit == model.y_unit


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


# ==========================================
# Tests for FastCutoffPowerlawPyTorch
# ==========================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def make_function(K=1.0, piv=1.0, index=-2.0, xc=10.0):
    f = FastCutoffPowerlawPyTorch()
    f.K = K
    f.piv = piv
    f.index = index
    f.xc = xc
    return f
 
 
def analytic_cutoff_powerlaw(x, K, piv, index, xc):
    return K * np.power(np.asarray(x, dtype=float) / piv, index) * np.exp(
        -np.asarray(x, dtype=float) / xc
    )
 
 
# ---------------------------------------------------------------------------
# Parameter defaults / metadata (from the docstring YAML block)
# ---------------------------------------------------------------------------
 
def test_default_parameter_values():
    f = FastCutoffPowerlawPyTorch()
    assert pytest.approx(f.K.value) == 1.0
    assert pytest.approx(f.piv.value) == 1.0
    assert pytest.approx(f.index.value) == -2.0
    assert pytest.approx(f.xc.value) == 10.0
 
 
def test_piv_is_fixed_by_default():
    f = FastCutoffPowerlawPyTorch()
    assert f.piv.free is False
 
 
def test_K_is_normalization():
    f = FastCutoffPowerlawPyTorch()
    assert f.K.is_normalization is True
 
 
def test_parameter_bounds():
    f = FastCutoffPowerlawPyTorch()
    assert f.K.min_value == pytest.approx(1e-30)
    assert f.K.max_value == pytest.approx(1e3)
    assert f.index.min_value == pytest.approx(-10)
    assert f.index.max_value == pytest.approx(10)
    assert f.xc.min_value == pytest.approx(1.0)
 
 
def test_devices_property_defaults_to_cpu():
    f = FastCutoffPowerlawPyTorch()
    assert f.devices.value == "cpu"
 
 
# ---------------------------------------------------------------------------
# _set_units
# ---------------------------------------------------------------------------
 
def test_set_units_assigns_expected_units():
    f = FastCutoffPowerlawPyTorch()
    x_unit = u.keV
    y_unit = 1 / (u.keV * u.cm ** 2 * u.s)
 
    f.set_units(x_unit, y_unit)
 
    assert f.index.unit == u.dimensionless_unscaled
    assert f.piv.unit == x_unit
    assert f.xc.unit == x_unit
    assert f.K.unit == y_unit
 
 
# ---------------------------------------------------------------------------
# evaluate: plain (non-Quantity) input
# ---------------------------------------------------------------------------
 
def test_evaluate_matches_analytic_formula():
    f = make_function(K=2.0, piv=1.0, index=-2.0, xc=10.0)
    x = np.array([1.0, 2.0, 5.0, 10.0])
 
    result = f.evaluate(x, K=2.0, piv=1.0, index=-2.0, xc=10.0)
    expected = analytic_cutoff_powerlaw(x, K=2.0, piv=1.0, index=-2.0, xc=10.0)
 
    # NOTE: evaluate() currently returns shape (N, 1) due to `.view(-1, 1)`
    assert result.shape == (len(x), 1)
    np.testing.assert_allclose(result.ravel(), expected, rtol=1e-6)
 
 
def test_evaluate_scalar_input():
    f = make_function()
    result = f.evaluate(5.0, K=1.0, piv=1.0, index=-2.0, xc=10.0)
    expected = analytic_cutoff_powerlaw(5.0, K=1.0, piv=1.0, index=-2.0, xc=10.0)
    np.testing.assert_allclose(np.ravel(result), np.ravel(expected), rtol=1e-6)
 
 
def test_evaluate_returns_numpy_array():
    f = make_function()
    result = f.evaluate(np.array([1.0, 2.0]), K=1.0, piv=1.0, index=-2.0, xc=10.0)
    assert isinstance(result, np.ndarray)
 
 
def test_evaluate_no_units_applied_when_plain_input():
    # When x is not an astropy Quantity, the result should be a bare
    # array (unit_ == 1.0 branch), not multiplied by any astropy unit.
    f = make_function()
    result = f.evaluate(np.array([1.0, 2.0]), K=1.0, piv=1.0, index=-2.0, xc=10.0)
    assert not hasattr(result, "unit")
 
 
# ---------------------------------------------------------------------------
# evaluate: Quantity input
# ---------------------------------------------------------------------------
 
def test_evaluate_with_quantity_input_returns_quantity():
    f = make_function(K=1.0, piv=1.0, index=-2.0, xc=10.0)
    f.set_units(
        u.keV,
        1 / (u.keV * u.cm ** 2 * u.s),
    )

    x = np.array([1.0, 2.0, 5.0]) * u.keV
    result = f.evaluate(
        x,
        K=f.K.as_quantity,
        piv=f.piv.as_quantity,
        index=f.index.as_quantity,
        xc=f.xc.as_quantity,
    )
 
    assert hasattr(result, "unit")
    assert result.unit == f.y_unit
 
 
def test_evaluate_quantity_matches_plain_evaluation_numerically():
    f = make_function(K=3.0, piv=2.0, index=-1.5, xc=7.0)
    f.set_units(
        u.keV,
        1 / (u.keV * u.cm ** 2 * u.s),
    )
    
 
    x_val = np.array([1.0, 3.0, 6.0])
    x_q = x_val * u.keV
 
    plain_result = f.evaluate(x_val, K=3.0, piv=2.0, index=-1.5, xc=7.0)
    quantity_result = f.evaluate(
        x_q,
        K=f.K.as_quantity,
        piv=f.piv.as_quantity,
        index=f.index.as_quantity,
        xc=f.xc.as_quantity,
    )
 
    np.testing.assert_allclose(
        plain_result.ravel(), quantity_result.value.ravel(), rtol=1e-6
    )
 
 
# ---------------------------------------------------------------------------
# Physical / shape behavior
# ---------------------------------------------------------------------------
 
def test_value_at_pivot_equals_K_times_cutoff_term():
    f = make_function(K=5.0, piv=2.0, index=-2.0, xc=10.0)
    result = f.evaluate(np.array([2.0]), K=5.0, piv=2.0, index=-2.0, xc=10.0)
    expected = 5.0 * np.exp(-2.0 / 10.0)
    np.testing.assert_allclose(result.ravel()[0], expected, rtol=1e-6)
 
 
def test_decreasing_with_negative_index_before_cutoff_dominates():
    # For a steep negative index and a cutoff far away, flux should
    # decrease as x increases (power-law term dominates).
    f = make_function(K=1.0, piv=1.0, index=-2.0, xc=1e6)
    x = np.array([1.0, 2.0, 4.0])
    result = f.evaluate(x, K=1.0, piv=1.0, index=-2.0, xc=1e6).ravel()
    assert result[0] > result[1] > result[2]
 
 
def test_cutoff_suppresses_flux_well_above_xc():
    f = make_function(K=1.0, piv=1.0, index=0.0, xc=1.0)
    x = np.array([0.1, 1.0, 10.0, 50.0])
    result = f.evaluate(x, K=1.0, piv=1.0, index=0.0, xc=1.0).ravel()
    # Well beyond the cutoff energy, flux should be strongly suppressed
    assert result[-1] < result[0] * 1e-10
 
 
def test_result_is_non_negative():
    f = make_function(K=1.0, piv=1.0, index=-2.0, xc=10.0)
    x = np.linspace(0.5, 20, 10)
    result = f.evaluate(x, K=1.0, piv=1.0, index=-2.0, xc=10.0)
    assert np.all(result >= 0)
 
 
def test_output_length_matches_input_length():
    f = make_function()
    x = np.linspace(1, 100, 25)
    result = f.evaluate(x, K=1.0, piv=1.0, index=-2.0, xc=10.0)
    assert result.shape[0] == len(x)
    
        
# ==========================================
# Tests for FastSuperCutoffPowerlawPyTorch
# ==========================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def make_function(K=1.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0):
    f = FastSuperCutoffPowerlawPyTorch()
    f.K = K
    f.piv = piv
    f.index = index
    f.xc = xc
    f.gamma = gamma
    return f
 
 
def analytic_cutoff_powerlaw(x, K, piv, index, xc, gamma):
    return K * np.exp( index * np.log(x / piv) - np.pow(x / xc, gamma) )

 
# ---------------------------------------------------------------------------
# Parameter defaults / metadata (from the docstring YAML block)
# ---------------------------------------------------------------------------
 
def test_default_parameter_values():
    f = FastSuperCutoffPowerlawPyTorch()
    assert pytest.approx(f.K.value) == 1.0
    assert pytest.approx(f.piv.value) == 1.0
    assert pytest.approx(f.index.value) == -2.0
    assert pytest.approx(f.xc.value) == 10.0
    assert pytest.approx(f.gamma.value) == 1.0
 
def test_piv_is_fixed_by_default():
    f = FastSuperCutoffPowerlawPyTorch()
    assert f.piv.free is False
 
 
def test_K_is_normalization():
    f = FastSuperCutoffPowerlawPyTorch()
    assert f.K.is_normalization is True
 
 
def test_parameter_bounds():
    f = FastSuperCutoffPowerlawPyTorch()
    assert f.K.min_value == pytest.approx(1e-50)
    assert f.index.min_value == pytest.approx(-10)
    assert f.index.max_value == pytest.approx(10)
    assert f.xc.min_value == pytest.approx(1.0)
    assert f.gamma.min_value == pytest.approx(0.1)
    assert f.gamma.max_value == pytest.approx(10.0)

def test_devices_property_defaults_to_cpu():
    f = FastSuperCutoffPowerlawPyTorch()
    assert f.devices.value == "cpu"
 
 
# ---------------------------------------------------------------------------
# _set_units
# ---------------------------------------------------------------------------
 
def test_set_units_assigns_expected_units():
    f = FastSuperCutoffPowerlawPyTorch()
    x_unit = u.keV
    y_unit = 1 / (u.keV * u.cm ** 2 * u.s)
 
    f.set_units(x_unit, y_unit)
 
    assert f.index.unit == u.dimensionless_unscaled
    assert f.piv.unit == x_unit
    assert f.xc.unit == x_unit
    assert f.K.unit == y_unit
    assert f.gamma.unit == u.dimensionless_unscaled
 
# ---------------------------------------------------------------------------
# evaluate: plain (non-Quantity) input
# ---------------------------------------------------------------------------
 
def test_evaluate_matches_analytic_formula():
    f = make_function(K=2.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0)
    x = np.array([1.0, 2.0, 5.0, 10.0])
 
    result = f.evaluate(x, K=2.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0)
    expected = analytic_cutoff_powerlaw(x, K=2.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0)
 
    # NOTE: evaluate() currently returns shape (N, 1) due to `.view(-1, 1)`
    assert result.shape == (len(x), 1)
    np.testing.assert_allclose(result.ravel(), expected, rtol=1e-6)
 
 
def test_evaluate_scalar_input():
    f = make_function()
    result = f.evaluate(5.0, K=1.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0)
    expected = analytic_cutoff_powerlaw(5.0, K=1.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0)
    np.testing.assert_allclose(np.ravel(result), np.ravel(expected), rtol=1e-6)
 
 
def test_evaluate_returns_numpy_array():
    f = make_function()
    result = f.evaluate(np.array([1.0, 2.0]), K=1.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0)
    assert isinstance(result, np.ndarray)
 
 
def test_evaluate_no_units_applied_when_plain_input():
    # When x is not an astropy Quantity, the result should be a bare
    # array (unit_ == 1.0 branch), not multiplied by any astropy unit.
    f = make_function()
    result = f.evaluate(np.array([1.0, 2.0]), K=1.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0)
    assert not hasattr(result, "unit")
 
 
# ---------------------------------------------------------------------------
# evaluate: Quantity input
# ---------------------------------------------------------------------------
 
def test_evaluate_with_quantity_input_returns_quantity():
    f = make_function(K=1.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0)
    f.set_units(
        u.keV,
        1 / (u.keV * u.cm ** 2 * u.s),
    )

    x = np.array([1.0, 2.0, 5.0]) * u.keV
    result = f.evaluate(
        x,
        K=f.K.as_quantity,
        piv=f.piv.as_quantity,
        index=f.index.as_quantity,
        xc=f.xc.as_quantity,
        gamma=f.gamma.as_quantity
    )
 
    assert hasattr(result, "unit")
    assert result.unit == f.y_unit
 
 
def test_evaluate_quantity_matches_plain_evaluation_numerically():
    f = make_function(K=3.0, piv=2.0, index=-1.5, xc=7.0, gamma=2.0)
    f.set_units(
        u.keV,
        1 / (u.keV * u.cm ** 2 * u.s),
    )
    
 
    x_val = np.array([1.0, 3.0, 6.0])
    x_q = x_val * u.keV
 
    plain_result = f.evaluate(x_val, K=3.0, piv=2.0, index=-1.5, xc=7.0, gamma=2.0)
    quantity_result = f.evaluate(
        x_q,
        K=f.K.as_quantity,
        piv=f.piv.as_quantity,
        index=f.index.as_quantity,
        xc=f.xc.as_quantity,
        gamma=f.gamma.as_quantity
    )
 
    np.testing.assert_allclose(
        plain_result.ravel(), quantity_result.value.ravel(), rtol=1e-6
    )
 
 
# ---------------------------------------------------------------------------
# Physical / shape behavior
# ---------------------------------------------------------------------------
 
def test_value_at_pivot_equals_K_times_cutoff_term():
    f = make_function(K=5.0, piv=2.0, index=-2.0, xc=10.0, gamma=2.0)
    result = f.evaluate(np.array([2.0]), K=5.0, piv=2.0, index=-2.0, xc=10.0, gamma=2.0)
    expected = analytic_cutoff_powerlaw(np.array([2.0]), K=5.0, piv=2.0, index=-2.0, xc=10.0, gamma=2.0)
    np.testing.assert_allclose(result.ravel()[0], expected, rtol=1e-6)
 
 
def test_decreasing_with_negative_index_before_cutoff_dominates():
    # For a steep negative index and a cutoff far away, flux should
    # decrease as x increases (power-law term dominates).
    f = make_function(K=1.0, piv=1.0, index=-2.0, xc=1e6, gamma=2.0)
    x = np.array([1.0, 2.0, 4.0])
    result = f.evaluate(x, K=1.0, piv=1.0, index=-2.0, xc=1e6, gamma=2.0).ravel()
    assert result[0] > result[1] > result[2]
 
 
def test_cutoff_suppresses_flux_well_above_xc():
    f = make_function(K=1.0, piv=1.0, index=0.0, xc=1.0, gamma=2.0)
    x = np.array([0.1, 1.0, 10.0, 50.0])
    result = f.evaluate(x, K=1.0, piv=1.0, index=0.0, xc=1.0, gamma=2.0).ravel()
    # Well beyond the cutoff energy, flux should be strongly suppressed
    assert result[-1] < result[0] * 1e-10
 
 
def test_result_is_non_negative():
    f = make_function(K=1.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0)
    x = np.linspace(0.5, 20, 10)
    result = f.evaluate(x, K=1.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0)
    assert np.all(result >= 0)
 
 
def test_output_length_matches_input_length():
    f = make_function()
    x = np.linspace(1, 100, 25)
    result = f.evaluate(x, K=1.0, piv=1.0, index=-2.0, xc=10.0, gamma=2.0)
    assert result.shape[0] == len(x)
