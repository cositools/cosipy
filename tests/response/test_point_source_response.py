import numpy as np
from numpy import array_equal as arr_eq
from histpy import Histogram
from scoords import SpacecraftFrame

import astropy.units as u

from astropy.time import Time
from astromodels.core.polarization import LinearPolarization
from mhealpy import HealpixMap

from cosipy import test_data
from cosipy.response import FullDetectorResponse

from threeML import DiracDelta, Constant, Line, Quadratic, Cubic, Quartic
from threeML import StepFunction, StepFunctionUpper, GenericFunction

import pytest

# init/load
response_path = test_data.path/"test_full_detector_response.h5"

with FullDetectorResponse.open(response_path) as response:
    exposure_map = HealpixMap(base=response,
                                    unit=u.s,
                                    coordsys=SpacecraftFrame())

    ti = Time('1999-01-01T00:00:00.123456789')
    tf = Time('2010-01-01T00:00:00')
    dt = (tf-ti).to(u.s)

    exposure_map[:4] = dt/4
    psr = response.get_point_source_response(exposure_map = exposure_map)

# pol response
rsp_pol_path = test_data.path/"test_polarization_response.h5"
with FullDetectorResponse.open(rsp_pol_path, pa_convention='RelativeX') as response_pol:
    exposure_map = HealpixMap(base=response,
                                    unit=u.s,
                                    coordsys=SpacecraftFrame())

    ti = Time('1999-01-01T00:00:00.123456789')
    tf = Time('2010-01-01T00:00:00')
    dt = (tf-ti).to(u.s)

    exposure_map[:4] = dt/4
    psr_pol = response_pol.get_point_source_response(exposure_map = exposure_map)

def test_photon_energy_axis():
    assert psr.photon_energy_axis == psr.axes['Ei']

def test_get_expectation():
    # supported spectral functions
    ## see astromodels.functions.function.Function1D.[function_name]()
    ## normalization units make expectation have units of counts
    norm = 1 / (u.keV * u.cm**2 * u.s)
    ## Note: rtol is relative tolerance or relative error

    ## Constant
    const = Constant(k=1e-1)
    with pytest.raises(RuntimeError) as r_error: # w/o norm should error
        exp = psr.get_expectation(const)
    assert r_error.type is RuntimeError
    const.k.unit = norm
    exp = psr.get_expectation(const)
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 7.84210661e+12, rtol=1e-8)

    ## Line
    line = Line(a=1e-1, b=4e-5)
    line.a.unit, line.b.unit = norm, norm
    exp = psr.get_expectation(line)
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 1.87182461e+13, rtol=1e-8)

    ## Quadratic
    quad = Quadratic(a=1e-1, b=4e-5, c=1e-9)
    quad.a.unit, quad.b.unit, quad.c.unit = norm, norm, norm
    exp = psr.get_expectation(quad)
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 2.02183488e+13, rtol=1e-8)

    ## Cubic
    cubic = Cubic(a=1e-1, b=4e-5, c=1e-9, d=4e-13)
    cubic.a.unit, cubic.b.unit, cubic.c.unit, cubic.d.unit = norm, norm, norm, norm
    exp = psr.get_expectation(cubic)
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 2.429903e+13, rtol=1e-8)

    ## Quartic
    quartic = Quartic(a=1e-1, b=4e-5, c=1e-9, d=4e-13, e=1e-17)
    for param in quartic.parameters:
        getattr(quartic, param).unit = norm
    exp = psr.get_expectation(quartic)
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 2.50716288e+13, rtol=1e-8)

    ## StepFunction
    step = StepFunction(upper_bound=3e2, lower_bound=0, value=1)
    step.upper_bound.unit, step.lower_bound.unit, step.value.unit = norm, norm, norm
    exp = psr.get_expectation(step)
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 2.300714202190e+12, rtol=1e-8)

    ## StepFunctionUpper (same as above except bounds are not "free")
    step_upper = StepFunctionUpper(upper_bound=3e2, lower_bound=0, value=1)
    step_upper.upper_bound.unit, step_upper.lower_bound.unit, step_upper.value.unit = norm, norm, norm
    exp = psr.get_expectation(step_upper)
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 2.300714202190e+12, rtol=1e-8)

    ## DiracDelta
    delta = DiracDelta(value=1, zero_point=251.189)
    delta.value.unit = norm
    exp = psr.get_expectation(delta)
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 3.21285337e+10, rtol=1e-8)

    ## Test manual integration from scipy.integrate
    gen = GenericFunction()
    gen.set_function(lambda x: 1e-1) # like Constant()
    # before setting units should throw error
    with pytest.raises(RuntimeError) as r_error:
        exp = psr.get_expectation(gen)
    assert r_error.type is RuntimeError
    gen.k.unit = norm
    exp = psr.get_expectation(gen)
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 7.84210661e+12, rtol=1e-8)

    ## test polarization error when rsp does not have 'Pol' axis
    with pytest.raises(RuntimeError) as r_error:
        exp = psr.get_expectation(const, polarization=LinearPolarization(angle=180, degree=100))
    assert r_error.type is RuntimeError

    ## test when rsp does have 'Pol' axis
    const = Constant(k=1e-1)
    const.k.unit = norm

    ## get expectation with polarization
    exp = psr_pol.get_expectation(const, polarization=LinearPolarization(angle=180, degree=100))
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 7.56988247e+12, rtol=1e-8)

    # If polarization is not given, the result should be the same as PD=0
    exp_none = psr_pol.get_expectation(const)
    exp0 = psr_pol.get_expectation(const, polarization=LinearPolarization(angle=0, degree=0))
    assert exp0 == exp_none


def test_spectrum_unit_generic():

    from astromodels import Function1D, FunctionMeta
    import astropy.units as u
    from cosipy.response.functions import get_spectrum_unit

    # test that we can get the unit from a custom function that
    # has never been seen before, provided that the unit is assigned
    # to the K or k parameter

    class MyFunction(Function1D, metaclass=FunctionMeta):
        r"""
        description :
            a silly function
        parameters:
            K :
              desc: Normalization
              initial value : 1.0
              unit : keV
        """
        def _set_units(self, x_unit, y_unit):
            # this is never actually called during setup
            pass

        def evaluate(self, x, K):
            return K * x

    f = MyFunction()
    unit = get_spectrum_unit(f)
    assert unit == u.keV

    class MyFunction2(Function1D, metaclass=FunctionMeta):
        r"""
        description :
            another silly function
        parameters:
            k :
              desc: Normalization
              initial value : 1.0
              unit : keV
        """
        def _set_units(self, x_unit, y_unit):
            # this is never actually called during setup
            pass

        def evaluate(self, x, k):
            return k * x

    f = MyFunction()
    unit = get_spectrum_unit(f)
    assert unit == u.keV
