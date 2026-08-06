import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.constants import h

from cosipy.data_io import SpectrumFileProcessor
from cosipy.threeml.custom_functions import SpecFromDat


def make_processor(tmp_path, rows, **kwargs):
    input_file = tmp_path / "input.dat"
    np.savetxt(input_file, rows)
    processor = SpectrumFileProcessor(
        input_file,
        tmp_path / "output.dat",
        **kwargs,
    )
    processor.load_data()
    return processor


def test_keV_photon_flux_input_and_custom_columns(tmp_path):
    processor = make_processor(
        tmp_path,
        [
            [1, 10001, 9],
            [4, 10000, 3],
            [2, 100, 1],
            [7, 50, 8],
            [3, 1000, 2],
        ],
        energy_col=1,
        flux_col=0,
        convert_data=False,
    )

    data = processor.process_data()

    np.testing.assert_array_equal(
        data[processor.energy_label],
        [100, 1000, 10000],
    )
    np.testing.assert_array_equal(
        data[processor.flux_label],
        [2, 3, 4],
    )


def test_frequency_nufnu_conversion(tmp_path):
    energies = np.array([100.0, 1000.0, 10000.0]) * u.keV
    frequencies = (energies / h).to_value(u.Hz)
    nu_fnu = np.array([1e-9, 2e-9, 3e-9])
    processor = make_processor(
        tmp_path,
        np.column_stack([frequencies, nu_fnu]),
    )

    data = processor.process_data()

    expected_flux = (
        (nu_fnu * u.erg / (u.cm**2 * u.s)).to_value(
            u.keV / (u.cm**2 * u.s)
        )
        / energies.to_value(u.keV) ** 2
    )
    np.testing.assert_allclose(data[processor.energy_label], energies.value)
    np.testing.assert_allclose(data[processor.flux_label], expected_flux)


def test_integrated_flux_uses_sorted_left_bins(tmp_path):
    processor = make_processor(
        tmp_path,
        [[1000, 2], [10000, 3], [100, 1]],
        convert_data=False,
    )
    processor.process_data()

    expected = 1 * (1000 - 100) + 2 * (10000 - 1000)
    assert processor.integrate_flux() == pytest.approx(expected)


def test_output_is_readable_by_specfromdat(tmp_path):
    energies = np.array([100.0, 200.0, 1000.0, 10000.0])
    flux = np.array([1e-4, 2.5e-5, 1e-6, 1e-8])
    processor = make_processor(
        tmp_path,
        np.column_stack([energies, flux]),
        convert_data=False,
    )
    processor.process_data()
    processor.reformat_data()

    output_file = tmp_path / "output.dat"
    lines = output_file.read_text().splitlines()
    assert lines[:6] == [
        "-Ps photon spectrum file",
        "#",
        "# Format: DP <energy in keV> <differential photon flux [ph/cm2/s/keV]>",
        "",
        "IP LIN",
        "",
    ]
    assert lines[-1] == "EN"

    spectrum = SpecFromDat(dat=output_file)
    normalization = np.sum(flux * np.diff(energies, append=energies[-1]))
    np.testing.assert_allclose(spectrum.evaluate(energies, 1), flux / normalization)


def test_plot_uses_log_axes(tmp_path, monkeypatch):
    processor = make_processor(
        tmp_path,
        [[100, 1], [1000, 0.1]],
        convert_data=False,
    )
    processor.process_data()
    monkeypatch.setattr(plt, "show", lambda: None)

    axes = processor.plot_spectrum()

    assert axes.get_xscale() == "log"
    assert axes.get_yscale() == "log"
    plt.close(axes.figure)


def test_load_data_accepts_one_row_but_processing_requires_two(tmp_path):
    processor = make_processor(
        tmp_path,
        [[1000, 1]],
        convert_data=False,
    )

    assert processor.data.shape == (1, 2)
    with pytest.raises(ValueError, match="At least two points"):
        processor.process_data()


def test_process_data_requires_loaded_data(tmp_path):
    processor = SpectrumFileProcessor(
        tmp_path / "input.dat",
        tmp_path / "output.dat",
    )

    with pytest.raises(ValueError, match=r"load_data\(\)"):
        processor.process_data()


@pytest.mark.parametrize(
    "rows",
    [
        [[100, 1], [np.nan, 1]],
        [[0, 1], [1000, 1]],
        [[100, 1], [1000, -1]],
        [[100, 1], [100, 2]],
    ],
)
def test_rejects_values_that_cannot_make_a_valid_spectrum(tmp_path, rows):
    processor = make_processor(tmp_path, rows, convert_data=False)

    with pytest.raises(ValueError):
        processor.process_data()
