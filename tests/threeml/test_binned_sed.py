from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astromodels import (
    Constant,
    Cutoff_powerlaw,
    Model,
    PointSource,
    Powerlaw,
    clone_model,
)

from cosipy.response.integrals import get_integral_values
from cosipy.threeml import BinnedSED, freeze_binned_sed_bins


def _response(nbins=8):
    edges = np.geomspace(100.0, 10000.0, nbins + 1) * u.keV
    return SimpleNamespace(
        axes={"Ei": SimpleNamespace(edges=edges, nbins=nbins)}
    )


def _powerlaw(index=-1.7):
    shape = Powerlaw()
    shape.K.value = 3.0e-5
    shape.piv.value = 300.0
    shape.index.value = index
    return shape


def test_binned_sed_defaults_to_powerlaw_index_minus_two():
    spectrum = BinnedSED.from_response(_response())

    assert isinstance(spectrum.spectral_shape, Powerlaw)
    assert spectrum.spectral_shape.index.value == -2.0
    assert spectrum.spectral_shape.index.free is False
    for i in range(spectrum.n_bins):
        assert spectrum.bin_shape_parameters(i)["index"].value == -2.0
        assert all(not p.free for p in spectrum.bin_shape_parameters(i).values())
    assert "index" not in spectrum.parameters


def test_binned_sed_snapshots_and_freezes_spectral_shape():
    input_shape = _powerlaw()
    initial_fluxes = np.array([1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6])
    spectrum = BinnedSED.from_response(
        _response(),
        input_shape,
        ei_bin_indices=range(2, 6),
        initial_fluxes=initial_fluxes,
    )

    assert isinstance(spectrum, BinnedSED)
    assert spectrum.n_bins == 4
    assert spectrum._cosipy_ei_bin_indices == (2, 3, 4, 5)
    assert all(
        not parameter.free
        for parameter in spectrum.spectral_shape.parameters.values()
    )
    assert all(parameter.free for parameter in spectrum.normalizations)
    assert np.allclose(spectrum(spectrum.pivots), initial_fluxes)

    frozen_index = spectrum.spectral_shape.index.value
    input_shape.index.value = -3.0
    assert spectrum.spectral_shape.index.value == frozen_index


def test_binned_sed_uses_all_response_bins_by_default():
    response = _response(nbins=7)
    spectrum = BinnedSED.from_response(response, _powerlaw())

    assert spectrum.n_bins == 7
    assert spectrum._cosipy_ei_bin_indices == tuple(range(7))
    assert len(spectrum.normalizations) == 7


def test_binned_sed_zero_outside_range():
    spectrum = BinnedSED.from_response(_response(nbins=5), _powerlaw())
    edges = spectrum.bin_edges

    flux = spectrum(np.array([0.5 * edges[0], 2.0 * edges[-1]]))

    assert np.array_equal(flux, [0.0, 0.0])


def test_powerlaw_shape_has_expected_within_bin_values_and_integrals():
    response = _response(nbins=4)
    initial_fluxes = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    spectrum = BinnedSED.from_response(
        response,
        _powerlaw(index=-1.0),
        initial_fluxes=initial_fluxes,
    )
    edges = spectrum.bin_edges
    expected_integral = sum(
        k
        * np.sqrt(edges[i] * edges[i + 1])
        * np.log(edges[i + 1] / edges[i])
        for i, k in enumerate(initial_fluxes)
    )

    assert np.allclose(spectrum(spectrum.pivots), initial_fluxes)
    assert np.isclose(
        spectrum.integral(edges[0], edges[-1]),
        expected_integral,
    )


def test_cutoff_shape_retains_curvature_inside_each_bin():
    response = _response(nbins=4)
    shape = Cutoff_powerlaw()
    shape.K.value = 2.0e-5
    shape.piv.value = 200.0
    shape.index.value = -2.0
    shape.xc.value = 450.0
    initial_fluxes = np.array([1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6])
    spectrum = BinnedSED.from_response(
        response,
        shape,
        initial_fluxes=initial_fluxes,
    )

    energy = 0.75 * spectrum.bin_edges[2] + 0.25 * spectrum.bin_edges[3]
    expected = initial_fluxes[2] * shape(energy) / shape(spectrum.pivots[2])
    assert np.isclose(spectrum(energy), expected)


def test_binned_sed_supports_units():
    spectrum = BinnedSED.from_response(_response(nbins=3), _powerlaw())
    flux_unit = 1.0 / (u.keV * u.cm**2 * u.s)
    spectrum.set_units(u.keV, flux_unit)

    values = spectrum(spectrum.pivots * u.keV)
    assert values.unit == flux_unit
    assert np.allclose(
        values.value,
        [parameter.value for parameter in spectrum.normalizations],
    )


def test_binned_sed_rejects_zero_pivot_shape():
    shape = Constant()
    shape.k.value = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        BinnedSED.from_response(_response(), shape)


def test_existing_bin_freezing_utility_accepts_shape_driven_sed():
    spectrum = BinnedSED.from_response(_response(), _powerlaw())

    frozen = freeze_binned_sed_bins(spectrum, [1, 3])

    assert frozen == (1, 3)
    assert spectrum.K1.value == 0.0
    assert spectrum.K1.free is False
    assert spectrum.K3.value == 0.0
    assert spectrum.K3.free is False


def test_astromodels_model_exposes_only_bin_normalizations_as_free():
    spectrum = BinnedSED.from_response(_response(), _powerlaw())
    model = Model(
        PointSource("test_source", l=0.0, b=0.0, spectral_shape=spectrum)
    )

    free_paths = tuple(model.free_parameters)
    assert len(free_paths) == spectrum.n_bins
    assert all(path.rsplit(".", 1)[-1].startswith("K") for path in free_paths)
    assert "index" not in spectrum.parameters


def test_astromodels_model_clone_restores_frozen_spectral_shape():
    shape = _powerlaw(index=-1.35)
    spectrum = BinnedSED.from_response(_response(), shape)
    model = Model(
        PointSource("test_source", l=0.0, b=0.0, spectral_shape=spectrum)
    )

    cloned_model = clone_model(model)
    cloned_spectrum = cloned_model.test_source.spectrum.main.shape

    assert cloned_spectrum.spectral_shape.index.value == -1.35
    assert all(
        not parameter.free
        for parameter in cloned_spectrum.spectral_shape.parameters.values()
    )
    assert np.allclose(
        cloned_spectrum(cloned_spectrum.pivots),
        [parameter.value for parameter in cloned_spectrum.normalizations],
    )


def test_response_integrals_match_direct_binned_sed_integrals():
    response = _response(nbins=5)
    spectrum = BinnedSED.from_response(
        response,
        _powerlaw(index=-1.35),
        initial_fluxes=np.geomspace(1.0e-7, 5.0e-6, 5),
    )
    values = get_integral_values(spectrum, response.axes["Ei"].edges.value)
    expected = np.asarray(
        [
            spectrum.integral(low, high)
            for low, high in zip(spectrum.bin_edges[:-1], spectrum.bin_edges[1:])
        ]
    )

    assert np.allclose(values, expected)


def test_shape_parameter_is_registered_and_free_only_in_selected_bin():
    shape = Cutoff_powerlaw()
    shape.index.value = -2
    shape.xc.value = 600
    spectrum = BinnedSED.from_response(_response(4), shape)
    parameters = spectrum.bin_shape_parameters(1)
    assert parameters["xc"] is spectrum.shape1_xc
    parameters["xc"].free = True
    parameters["index"].free = True
    model = Model(PointSource("source", l=0, b=0, spectral_shape=spectrum))
    assert {p.rsplit(".", 1)[-1] for p in model.free_parameters} == {
        "K0", "K1", "K2", "K3", "shape1_xc", "shape1_index",
    }
    assert all(not p.free for p in spectrum.bin_shape_parameters(0).values())
    assert all(not p.free for p in spectrum.bin_shape_parameters(2).values())


@pytest.mark.parametrize("parameter,value", [("index", -1.2), ("xc", 900.)])
def test_changing_one_bin_shape_changes_only_its_values_and_integrals(parameter, value):
    shape = Cutoff_powerlaw()
    shape.xc.value = 500
    spectrum = BinnedSED.from_response(_response(3), shape)
    energy = spectrum.bin_edges[:-1] * 1.1
    before = spectrum(energy)
    integrals_before = get_integral_values(spectrum, spectrum.bin_edges)
    spectrum.bin_shape_parameters(1)[parameter].value = value
    after = spectrum(energy)
    integrals_after = get_integral_values(spectrum, spectrum.bin_edges)
    np.testing.assert_array_equal(before[[0, 2]], after[[0, 2]])
    np.testing.assert_array_equal(integrals_before[[0, 2]], integrals_after[[0, 2]])
    assert not np.isclose(before[1], after[1], rtol=1e-5, atol=0)
    assert not np.isclose(integrals_before[1], integrals_after[1], rtol=1e-5, atol=0)
    np.testing.assert_allclose(spectrum(spectrum.pivots), [p.value for p in spectrum.normalizations])
    changed_shape = spectrum.bin_spectral_shape(1)
    expected = spectrum.K1.value * changed_shape(energy[1]) / changed_shape(spectrum.pivots[1])
    np.testing.assert_allclose(after[1], expected)
    np.testing.assert_allclose(
        integrals_after,
        get_integral_values(spectrum, spectrum.bin_edges, force_quad=True),
        rtol=1e-6,
    )
    # The caller's shape and the original reference remain untouched.
    assert spectrum.spectral_shape.parameters[parameter].value == shape.parameters[parameter].value
    changed_shape.parameters[parameter].value = shape.parameters[parameter].value
    assert spectrum.bin_shape_parameters(1)[parameter].value == pytest.approx(value)


def test_per_bin_parameters_survive_model_clone_with_units_and_bounds():
    shape = Cutoff_powerlaw()
    shape.xc.value = 700
    shape.xc.bounds = (100, 2000)
    shape.xc.delta = 10
    spectrum = BinnedSED.from_response(_response(4), shape, ei_bin_indices=[1, 2])
    parameter = spectrum.bin_shape_parameters(1)["xc"]
    assert parameter.bounds == shape.xc.bounds
    assert parameter.delta == shape.xc.delta
    parameter.bounds = (100, 5000)
    parameter.value = 3000  # Outside the original reference bounds.
    parameter.free = True
    model = Model(PointSource("source", l=0, b=0, spectral_shape=spectrum))
    assert parameter.unit == u.keV
    cloned = clone_model(model).source.spectrum.main.shape
    other = cloned.bin_shape_parameters(1)["xc"]
    assert other.free is True
    assert other.value == pytest.approx(3000)
    assert other.bounds == (100, 5000)
    assert other.unit == u.keV
    assert cloned.bin_shape_parameters(0)["xc"].free is False
    energy = np.geomspace(*spectrum.bin_edges[[0, -1]], 30)
    np.testing.assert_allclose(cloned(energy), spectrum(energy))
    np.testing.assert_allclose(cloned(energy * u.keV).value, spectrum(energy))
    np.testing.assert_allclose(
        get_integral_values(cloned, spectrum.bin_edges),
        get_integral_values(spectrum, spectrum.bin_edges),
    )
    other.value = 1500
    assert parameter.value == pytest.approx(3000)


def test_bin_shape_parameter_indices_are_local_and_checked():
    spectrum = BinnedSED.from_response(_response(4), ei_bin_indices=[2, 3])
    assert spectrum.bin_shape_parameters(1)["index"] is spectrum.shape1_index
    for index in (-1, 2):
        with pytest.raises(IndexError, match="SED bin index"):
            spectrum.bin_shape_parameters(index)
    with pytest.raises(TypeError):
        spectrum.bin_shape_parameters(0.5)
    with pytest.raises(KeyError):
        spectrum.bin_shape_parameters(0)["xc"]


@pytest.mark.parametrize("family", ["powerlaw", "cutoff_powerlaw"])
def test_threeml_minuit_fits_a_selected_bin_shape_parameter(family):
    from threeML import DataList, JointLikelihood
    from threeML.plugin_prototype import PluginPrototype

    if family == "powerlaw":
        spectrum = BinnedSED.from_response(_response(2))
        parameter_name, truth_value = "index", -1.3
    else:
        shape = Cutoff_powerlaw()
        shape.index.value = -2
        shape.xc.value = 700
        spectrum = BinnedSED.from_response(_response(2), shape)
        parameter_name, truth_value = "xc", 400
    # A small spectral likelihood tests optimizer plumbing, without external
    # response data. It does not test identifiability in the COSI response.
    x = np.geomspace(110, 900, 20)
    truth_shape = spectrum.bin_spectral_shape(0)
    truth_shape.parameters[parameter_name].value = truth_value
    truth = 2e-6 * truth_shape(x) / truth_shape(spectrum.pivots[0])
    unchanged_value = spectrum.bin_shape_parameters(1)[parameter_name].value

    class SpectralData(PluginPrototype):
        def __init__(self):
            super().__init__("spectral_data", {})

        def set_model(self, model):
            self.model = model

        def get_log_like(self):
            prediction = self.model.source.spectrum.main.shape(x)
            return -0.5 * np.sum(((prediction - truth) / (0.02 * truth)) ** 2)

        def inner_fit(self):
            return self.get_log_like()

        def get_number_of_data_points(self):
            return len(x)

    spectrum.K1.free = False
    spectrum.bin_shape_parameters(0)[parameter_name].free = True
    model = Model(PointSource("source", l=0, b=0, spectral_shape=spectrum))
    fit = JointLikelihood(model, DataList(SpectralData()), verbose=False)
    fit.set_minimizer("minuit")
    fit.fit(quiet=True)
    assert spectrum.K0.value == pytest.approx(2e-6, rel=1e-3)
    assert spectrum.bin_shape_parameters(0)[parameter_name].value == pytest.approx(truth_value, rel=1e-3)
    assert spectrum.bin_shape_parameters(1)[parameter_name].value == unchanged_value


def test_composite_shape_parameters_and_priors_are_independent():
    from astromodels import Uniform_prior

    shape = _powerlaw(-1.2) + _powerlaw(-2.5)
    shape.parameters["index_1"].prior = Uniform_prior(lower_bound=-3, upper_bound=0)
    spectrum = BinnedSED.from_response(_response(2), shape)
    first = spectrum.bin_shape_parameters(0)["index_1"]
    second = spectrum.bin_shape_parameters(1)["index_1"]
    assert first.prior is not second.prior
    assert first.prior is not shape.parameters["index_1"].prior
    before = spectrum([150, 1500])
    first.value = -1.8
    first.free = True
    after = spectrum([150, 1500])
    assert before[0] != after[0]
    assert before[1] == after[1]
    model = Model(PointSource("source", l=0, b=0, spectral_shape=spectrum))
    cloned = clone_model(model).source.spectrum.main.shape
    assert cloned.bin_shape_parameters(0)["index_1"].has_prior()
    assert cloned.bin_shape_parameters(0)["index_1"].free
    np.testing.assert_allclose(cloned([150, 1500]), after)


def test_shape_parameters_preserve_transformations_in_cached_classes():
    shape = Cutoff_powerlaw()
    spectrum = BinnedSED.from_response(_response(2), shape)
    assert type(spectrum.shape0_xc.transformation) is type(shape.xc.transformation)
    shape.xc.remove_transformation()
    untransformed = BinnedSED.from_response(_response(2), shape)
    assert untransformed.shape0_xc.transformation is None


def test_per_bin_shape_changes_can_be_compensated_by_normalization():
    # A response-aligned SED bin supplies one integrated flux to the folding
    # operation. This degeneracy is why shape parameters are fixed by default.
    spectrum = BinnedSED.from_response(_response(3))
    original_flux = get_integral_values(spectrum, spectrum.bin_edges)
    spectrum.bin_shape_parameters(1)["index"].value = -1.1
    new_flux = get_integral_values(spectrum, spectrum.bin_edges)
    spectrum.K1.value *= original_flux[1] / new_flux[1]
    np.testing.assert_allclose(
        get_integral_values(spectrum, spectrum.bin_edges), original_flux, rtol=1e-12,
    )


def test_linked_input_shape_is_snapshotted_before_freeing_bin_parameter():
    from astromodels import Line

    reference = _powerlaw(-1.4)
    shape = _powerlaw()
    shape.index.add_auxiliary_variable(reference.index, Line(a=0, b=1))
    spectrum = BinnedSED.from_response(_response(2), shape)
    assert spectrum.bin_shape_parameters(0)["index"].value == -1.4
    assert not spectrum.spectral_shape.index.has_auxiliary_variable
    before = spectrum([150, 1500])
    parameter = spectrum.bin_shape_parameters(0)["index"]
    parameter.free = True
    parameter.value = -2.3
    after = spectrum([150, 1500])
    assert after[0] != before[0]
    assert after[1] == before[1]
    assert shape.index.has_auxiliary_variable
    reference.index.value = -2.7
    assert spectrum.bin_shape_parameters(1)["index"].value == -1.4


def test_per_bin_cutoff_values_follow_energy_unit_conversion():
    shape = Cutoff_powerlaw()
    shape.xc.value = 800
    spectrum = BinnedSED.from_response(_response(2), shape)
    flux_unit = u.keV**-1 * u.cm**-2 * u.s**-1
    spectrum.set_units(u.keV, flux_unit)
    spectrum.bin_shape_parameters(1)["xc"].value = 1200
    energy = np.array([150, 1500]) * u.keV
    before = spectrum(energy)
    spectrum.set_units(u.MeV, u.MeV**-1 * u.cm**-2 * u.s**-1)
    np.testing.assert_allclose(spectrum(energy).to_value(flux_unit), before.value)
    assert spectrum.bin_shape_parameters(1)["xc"].unit == u.MeV
    assert spectrum.bin_shape_parameters(1)["xc"].value == pytest.approx(1.2)
