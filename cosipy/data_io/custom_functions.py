import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import h

class SpectrumFileProcessor:
    def __init__(self, input_file, reformatted_file, energy_col=0, flux_col=1, convert_data=True):
        """
        Initialize the SpectrumFileProcessor class with default columns for energy and flux.
        
        Parameters:
        - input_file: Path to the input .dat file.
        - reformatted_file: Path to save the reformatted data.
        - energy_col: Index of the column containing energy or frequency (default: 0).
        - flux_col: Index of the column containing flux (default: 1).
        - convert_data: Boolean flag indicating if conversion is needed (default: True).
        """
        self.input_file = input_file
        self.reformatted_file = reformatted_file
        self.data = None
        self.df_filtered = None
        self.energy_col = energy_col  # Default to column 0 for energy
        self.flux_col = flux_col      # Default to column 1 for flux
        self.convert_data = convert_data  # Whether conversion is necessary

    def load_data(self):
        """Load the data from the .dat file."""
        try:
            self.data = np.loadtxt(self.input_file)
            print(f"Data loaded successfully from {self.input_file}")
        except Exception as e:
            print(f"Error loading file {self.input_file}: {e}")
            raise

    def process_data(self):
        """Process the data: Convert frequency to energy in keV and flux to ph/cm²/sec/keV if needed."""
        if self.convert_data:
            # If conversion is required
            energy_hz = self.data[:, self.energy_col] * u.Hz   # Column defined by user
            flux_ergs = self.data[:, self.flux_col] * u.erg / (u.cm**2 * u.s)  # Column defined by user
            
            # Convert frequency to energy in keV
            energy_keV = (h * energy_hz).to(u.keV)

            # Convert flux from ergs to keV
            flux_keV = flux_ergs.to(u.keV / (u.cm**2 * u.s))

            # Convert flux to ph/cm²/sec/keV
            flux_ph = (flux_keV / energy_keV**2)
        
        else:
            # If data is already in keV, assume first column is energy (keV) and second column is flux
            energy_keV = self.data[:, self.energy_col] * u.keV  # Directly use energy in keV
            flux_ph = self.data[:, self.flux_col] * u.ph / (u.cm**2 * u.s * u.keV)  # Directly use flux in ph/cm²/sec/keV

        # Create a DataFrame to store the data
        df = pd.DataFrame({
            'Energy (keV)': energy_keV.value,
            'Flux (ph/cm²/sec/keV)': flux_ph.value
        })

        # Filter out rows with energy less than 100 keV and more than 10000 keV
        self.df_filtered = df[(df['Energy (keV)'] >= 100) & (df['Energy (keV)'] <= 10000)]
        return self.df_filtered

    def integrate_flux(self):
        """
        Compute the total flux using the sum of flux multiplied by energy bin widths.
        
        Returns:
        - K: The total flux.
        """
        if self.df_filtered is None or self.df_filtered.empty:
            raise ValueError("No data to integrate. Please run process_data() first.")
        
        # Get energy and flux from the processed data
        energy = self.df_filtered['Energy (keV)'].values
        flux = self.df_filtered['Flux (ph/cm²/sec/keV)'].values

        # Calculate the energy bin widths
        ewidths = np.diff(energy, append=energy[-1])

        # Calculate the current total flux (integral of the spectrum)
        K = np.sum(flux * ewidths)

        print(f"Calculated Normalization Constant K: {K}")
        
        return K

    def plot_spectrum(self):
        """Generate a plot of energy vs flux with a log-log scale."""
        if self.df_filtered is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(self.df_filtered['Energy (keV)'], self.df_filtered['Flux (ph/cm²/sec/keV)'], marker="o", linestyle="-")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Flux (ph/cm²/sec/keV)")
            plt.title("Energy Flux Data")
            plt.grid(True)
            plt.show()
        else:
            print("No data available for plotting. Please run process_data first.")

    def reformat_data(self):
        """Reformat the data for the photon spectrum file."""
        try:
            formatted_lines = []
            formatted_lines.append("-Ps photon spectrum file")
            formatted_lines.append("#")
            formatted_lines.append("# Format: DP <energy in keV> <shape of O-Ps [XX/keV]>")
            formatted_lines.append("")
            formatted_lines.append("IP LIN")
            formatted_lines.append("")
            
            # Iterate through the DataFrame to format each line as required
            for index, row in self.df_filtered.iterrows():
                energy = row['Energy (keV)']
                flux = row['Flux (ph/cm²/sec/keV)']
                formatted_line = f"DP\t{energy:.5e}\t{flux:.5e}"  # Add DP and reformat the line
                formatted_lines.append(formatted_line)
            
            # Append the closing 'EN' line
            formatted_lines.append("EN")
            
            # Save the new formatted data to a file
            with open(self.reformatted_file, 'w') as f:
                f.write("\n".join(formatted_lines))
            
            print(f"Reformatted data saved to {self.reformatted_file}")
        
        except Exception as e:
            print(f"Error during reformatting: {e}")
            raise