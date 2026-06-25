import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.constants import h


class SpectrumFileProcessor:
    """Prepare a simple two-column spectrum for the source injector tutorials.

    This helper intentionally supports only two input conventions:

    convert_data=True: frequency in Hz and nuFnu in erg/cm2/s.
    convert_data=False: energy in keV and dN/dE in ph/cm2/s/keV.

    Other formats and units must be converted by the user before using this class. The output is limited to 100-10000 keV and written in the format expected by cosipy.threeml.custom_functions.SpecFromDat.

    Examples
    --------
    1. Convert a two-column file with frequency_Hz and nuFnu_erg_cm2_s columns:

        processor = SpectrumFileProcessor(
            "raw_spectrum.txt",
            "source_injector_spectrum.dat",
            energy_col=0,
            flux_col=1,
            convert_data=True,
        )
        processor.load_data()
        processor.process_data()
        K = processor.integrate_flux()
        processor.reformat_data()

    2. Reformat a file that already has energy_keV and dnde_ph_cm2_s_keV columns:

        processor = SpectrumFileProcessor(
            "raw_spectrum.txt",
            "source_injector_spectrum.dat",
            convert_data=False,
        )
        processor.load_data()
        processor.process_data()
        processor.reformat_data()
    """

    energy_label = "Energy (keV)"
    flux_label = "Flux (ph/cm²/sec/keV)"

    def __init__(
        self,
        input_file,
        reformatted_file,
        energy_col=0,
        flux_col=1,
        convert_data=True,
    ):
        self.input_file = input_file
        self.reformatted_file = reformatted_file
        self.energy_col = energy_col
        self.flux_col = flux_col
        self.convert_data = convert_data
        self.data = None
        self.df_filtered = None

    def load_data(self):
        """Load a whitespace-delimited numeric text file."""
        self.data = np.loadtxt(self.input_file, ndmin=2)
        self.df_filtered = None
        return self.data

    def process_data(self):
        """Convert the selected columns and keep the source-injector range."""
        if self.data is None:
            raise ValueError("Run load_data() before process_data().")

        try:
            energy = self.data[:, self.energy_col]
            flux = self.data[:, self.flux_col]
        except (IndexError, TypeError) as error:
            raise ValueError(
                "energy_col and flux_col must select valid columns."
            ) from error

        if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(flux)):
            raise ValueError("Energy and flux values must be finite.")
        if np.any(energy <= 0) or np.any(flux < 0):
            raise ValueError("Energy must be positive and flux must be non-negative.")

        if self.convert_data:
            energy_keV = (h * energy * u.Hz).to_value(u.keV)
            energy_flux = (flux * u.erg / (u.cm**2 * u.s)).to_value(
                u.keV / (u.cm**2 * u.s)
            )
            photon_flux = energy_flux / energy_keV**2
        else:
            energy_keV = energy
            photon_flux = flux

        in_range = (energy_keV >= 100) & (energy_keV <= 10000)
        energy_keV = energy_keV[in_range]
        photon_flux = photon_flux[in_range]

        order = np.argsort(energy_keV)
        energy_keV = energy_keV[order]
        photon_flux = photon_flux[order]

        if len(energy_keV) < 2:
            raise ValueError("At least two points are required from 100 to 10000 keV.")
        if np.any(np.diff(energy_keV) == 0):
            raise ValueError("Energy values must be unique.")

        self.df_filtered = pd.DataFrame(
            {
                self.energy_label: energy_keV,
                self.flux_label: photon_flux,
            }
        )
        return self.df_filtered

    def _processed_data(self):
        if self.df_filtered is None:
            raise ValueError("Run process_data() first.")
        return self.df_filtered

    def integrate_flux(self):
        """Return the normalization used by ``SpecFromDat``."""
        data = self._processed_data()
        energy = data[self.energy_label].to_numpy()
        flux = data[self.flux_label].to_numpy()
        bin_widths = np.diff(energy, append=energy[-1])
        return np.sum(flux * bin_widths)

    def plot_spectrum(self):
        """Plot the processed spectrum."""
        data = self._processed_data()
        _, axes = plt.subplots(figsize=(10, 6))
        axes.plot(data[self.energy_label], data[self.flux_label], marker="o")
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlabel("Energy (keV)")
        axes.set_ylabel("Flux (ph/cm2/sec/keV)")
        axes.grid(True)
        plt.show()
        return axes

    def reformat_data(self):
        """Write a MEGAlib-style file that can be read by ``SpecFromDat``."""
        data = self._processed_data()
        lines = [
            "-Ps photon spectrum file",
            "#",
            "# Format: DP <energy in keV> <differential photon flux [ph/cm2/s/keV]>",
            "",
            "IP LIN",
            "",
        ]

        for energy, flux in data[[self.energy_label, self.flux_label]].to_numpy():
            lines.append(f"DP\t{energy:.5e}\t{flux:.5e}")

        lines.append("EN")
        with open(self.reformatted_file, "w") as output:
            output.write("\n".join(lines))
