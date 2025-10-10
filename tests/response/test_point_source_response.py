import numpy as np
from numpy import array_equal as arr_eq
from histpy import Histogram, Axes, Axis
from scoords import SpacecraftFrame
from astropy.coordinates import SkyCoord
import astropy.units as u
import h5py as h5
from astropy.time import Time
from astromodels.core.polarization import LinearPolarization
from mhealpy import HealpixBase, HealpixMap

from cosipy import test_data
from cosipy.response.FullDetectorResponse import cosi_response
from cosipy.response import PointSourceResponse, FullDetectorResponse

from threeML import DiracDelta, Constant, Line, Quadratic, Cubic, Quartic 
from threeML import StepFunction, StepFunctionUpper, GenericFunction
from cosipy.threeml.custom_functions import SpecFromDat

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
    assert np.isclose(np.sum(exp.contents), 2.3038894e+12, rtol=1e-8)
    
    ## StepFunctionUpper (same as above except bounds are not "free")
    step_upper = StepFunctionUpper(upper_bound=3e2, lower_bound=0, value=1)
    step_upper.upper_bound.unit, step_upper.lower_bound.unit, step_upper.value.unit = norm, norm, norm
    exp = psr.get_expectation(step_upper)
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 2.3038894e+12, rtol=1e-8)

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
    ## throw an error if polarization is not given
    with pytest.raises(RuntimeError) as r_error:
        exp = psr_pol.get_expectation(const)
    assert r_error.type is RuntimeError
    ## get expectation with polarization
    exp = psr_pol.get_expectation(const, polarization=LinearPolarization(angle=180, degree=100))
    assert isinstance(exp, Histogram)
    assert np.isclose(np.sum(exp.contents), 6.30823539e+11, rtol=1e-8)