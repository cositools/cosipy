import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class PlotPulseProfile:
    """
    Generates a diagnostic 3-panel visualization for pulsar timing analysis.
    
    This class creates a standardized figure containing:
    1. An Integrated Pulse Profile (histogram of counts vs. phase).
    2. A Phaseogram (2D heatmap of phase vs. time to check signal stability).
    3. A Cumulative Significance Test ($Z^2_2$ statistic) to track signal growth.

    The implementation is optimized for performance using NumPy vectorization 
    to handle large event lists directly from FITS tables or NumPy arrays.
    """

    def __init__(self, data, n_bins=50, n_time_bins=50):
        """
        Initializes the plotter with event data and binning configurations.

        Args:
            data (astropy.table.Table, FITS_rec, or np.ndarray): Structured data 
                containing at least a 'PULSE_PHASE' column and a time column 
                ('TimeTags' or 'TIME').
            n_bins (int): Number of phase bins for the profile and phaseogram. 
                Defaults to 50.
            n_time_bins (int): Number of vertical time intervals for the 
                phaseogram. Defaults to 50.
        """
        self.n_bins = n_bins
        self.n_time_bins = n_time_bins
        
        # --- ROBUST COLUMN DETECTION ---
        # Checks both .dtype.names (NumPy) and .names (Astropy/FITS)
        if hasattr(data, "dtype") and getattr(data.dtype, "names", None) is not None:
            cols = list(data.dtype.names)
        elif hasattr(data, "names"):
            cols = list(data.names)
        else:
            cols = []

        # --- DATA EXTRACTION ---
        if 'PULSE_PHASE' in cols:
            self.phases = np.asarray(data['PULSE_PHASE'])
        else:
            print("Error: 'PULSE_PHASE' column not found in data.")
            self.phases = np.array([])

        if 'TimeTags' in cols:
            self.times = np.asarray(data['TimeTags'])
        elif 'TIME' in cols:
            self.times = np.asarray(data['TIME'])
        else:
            # Fallback if no time column is found
            self.times = np.zeros(len(self.phases))

    def plot(self, t_start_met=None):
        """
        Renders the 3-panel pulsar diagnostic plot.

        Calculates an integrated profile over two cycles for visual continuity, 
        generates a 2D phaseogram histogram, and computes the cumulative 
        $Z^2_2$ (H-test variant) significance over time.

        Args:
            t_start_met (float, optional): The reference Mission Elapsed Time (MET) 
                to use as T=0. If None, the minimum time in the dataset is used.

        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        if len(self.phases) == 0:
            print("No valid events to plot.")
            return None

        # Relative Time Calculation
        if t_start_met is None:
            t_start_met = np.min(self.times)
        
        t_elapsed = self.times - t_start_met
        duration = np.max(t_elapsed)
        if duration <= 0: duration = 1.0

        # --- Plotting Setup ---
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1])
        
        ax_prof = fig.add_subplot(gs[0, 0])
        ax_htest = fig.add_subplot(gs[1, 0])
        ax_phaseogram = fig.add_subplot(gs[:, 1])

        # --- Panel 1: Integrated Profile (Top Left) ---
        counts, edges = np.histogram(self.phases, bins=self.n_bins, range=(0, 1))
        centers = (edges[:-1] + edges[1:]) / 2
        
        # 2-cycle plot for better visualization of peak wrap-around
        x_2cycle = np.concatenate([centers, centers + 1])
        y_2cycle = np.concatenate([counts, counts])
        
        ax_prof.step(x_2cycle, y_2cycle, where='mid', color='rebeccapurple', lw=2)
        ax_prof.set_xlim(0, 2)
        ax_prof.set_ylabel("Counts")
        ax_prof.set_xlabel("Pulse Phase")
        ax_prof.set_title(f"Integrated Profile (N={len(self.phases)})")
        ax_prof.grid(alpha=0.3)

        # --- Panel 2: Phaseogram (Right) ---
        
        h2d, xedges, yedges = np.histogram2d(
            self.phases, t_elapsed, 
            bins=[self.n_bins, self.n_time_bins], 
            range=[[0, 1], [0, duration]]
        )
        
        im = ax_phaseogram.imshow(h2d.T, origin='lower', aspect='auto', 
                                  extent=[0, 1, 0, duration], 
                                  cmap='viridis', interpolation='nearest')
        ax_phaseogram.set_xlabel("Pulse Phase")
        ax_phaseogram.set_ylabel("Time since start (s)")
        ax_phaseogram.set_title("Phaseogram")
        plt.colorbar(im, ax=ax_phaseogram, label="Counts/bin")

        # --- Panel 3: Significance (Bottom Left) ---
        if len(t_elapsed) > 1:
            sort_idx = np.argsort(t_elapsed)
            sorted_phases = self.phases[sort_idx] * 2 * np.pi 
            sorted_times = t_elapsed[sort_idx]
            
            # Cumulative Z^2_2 statistic (2 harmonics)   
            #TODO: We will make the maximum harmonic number a parameter in the future
            ns = np.arange(1, len(sorted_phases) + 1)
            cum_cos1 = np.cumsum(np.cos(sorted_phases))
            cum_sin1 = np.cumsum(np.sin(sorted_phases))
            cum_cos2 = np.cumsum(np.cos(2 * sorted_phases))
            cum_sin2 = np.cumsum(np.sin(2 * sorted_phases))
            
            z2_stats = (2.0 / ns) * (cum_cos1**2 + cum_sin1**2 + cum_cos2**2 + cum_sin2**2)
            
            step = max(1, len(z2_stats) // 2000)
            ax_htest.plot(sorted_times[::step], z2_stats[::step], '-', color='rebeccapurple', lw=1.5)
        
        ax_htest.set_xlabel("Time since start (s)")
        ax_htest.set_ylabel(r"Significance ($Z^2_2$)")
        ax_htest.set_title("Detection Significance")
        ax_htest.grid(alpha=0.3)

        plt.tight_layout()
        return fig
