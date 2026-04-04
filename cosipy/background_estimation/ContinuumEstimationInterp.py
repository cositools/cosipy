import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from histpy import Histogram
from cosipy.response import DetectorResponse

from mhealpy import HealpixMap, HealpixBase
import time
import logging

logger = logging.getLogger(__name__)

class ContinuumEstimationInterp:
    """
    Continuum background estimation using simple wrapped 1D interpolation
    in HEALPix NESTED index space.

    - Uses the same data loading + PSR-based masking as ContinuumEstimationNN.
    - Replaces NN inpainting with:
        For each masked pixel i:
          * search left (wrapping) for first unmasked pixel -> L
          * search right (wrapping) for first unmasked pixel -> R
          * fill(i) = 0.5*(L + R)
        Edge cases handled with fallbacks.

    This class can be ran without having pytorch installed.
    """

    @staticmethod
    def load_projected_data(h5_file):

        """Loads h5 data file and projects to Em, Phi, and Psichi.
        This is the input file with the total data used to estimate
        the background.

        Parameters
        ----------
        h5_file : str
            Full path to background h5 file.

        Returns
        -------
        projected : histogram
            Histogram object projected onto Em, Phi, and PsiChi.
        data : array
            Contents of histogram given as an array.
        """

        hist = Histogram.open(h5_file)
        projected = hist.project(['Em', 'Phi', 'PsiChi'])
        data = projected.contents.todense()

        return projected, data

    @staticmethod
    def load_projected_psr(psr_file):

        """Loads precomputed point source response and projects to Em, Phi, and Psichi.

        Parameters
        ----------
        psr_file : str
            Full path to precomputed point source response file.

        Returns
        -------
        projected : histogram
            Histogram object projected onto Em, Phi, and PsiChi.
        data : array
            Contents of histogram given as an array.
        """

        logger.info("...loading the pre-computed point source response ...")
        psr = DetectorResponse.open(psr_file)
        logger.info("--> done")

        projected = psr.project(['Em', 'Phi', 'PsiChi'])
        data = projected.contents.value

        return projected, data

    @staticmethod
    def load_projected_model(model_file):

        """Loads model file and projects to Em, Phi, and Psichi.

        Parameters
        ----------
        model_file : str
            Full path to model h5 file.

        Returns
        -------
        projected : histogram
            Histogram object projected onto Em, Phi, and PsiChi.
        data : array
            Contents of histogram object given as an array.
        """

        hist = Histogram.open(model_file)
        projected = hist.project(['Em', 'Phi', 'PsiChi'])
        data = projected.contents.todense()

        return projected, data

    @staticmethod
    def mask_from_cumdist_vectorized(psr_map, containment=0.4):

        """Masks point source cone in CDS based on PSR and ARM.

        Parameters
        ----------
        psr_map : array
            Point source response array, i.e. contents
            of histogram from load_projected_psr method.
        containment : float
            Fraction of ARM to mask (default is 0.4).

        Returns
        -------
        mask : array
            Boolean array defining mask.
        """

        psr_norm = psr_map / np.sum(psr_map, axis=-1, keepdims=True)
        sort_idx = np.argsort(psr_norm, axis=-1)[..., ::-1]
        sorted_vals = np.take_along_axis(psr_norm, sort_idx, axis=-1)
        cumsum_vals = np.cumsum(sorted_vals, axis=-1)
        mask_sorted = (cumsum_vals < containment).astype(float)
        mask = np.empty_like(mask_sorted)
        np.put_along_axis(mask, sort_idx, mask_sorted, axis=-1)
        # Invert so brightest pixels = 0 (mask), background = 1 (keep)
        mask = 1 - mask

        return mask

    @staticmethod
    def save_inpainted_histpy(projected_hist, inpainted_maps, output_file="inpainted.h5"):

        """Save results.

        projected_hist : histogram
            Histrogram object.
        inpainted_maps : array
            Inpainted background map.
        output_file : str, optional
            Name of output histogram file. Default is inpainted.h5.
        """

        hist = Histogram(projected_hist.axes, contents=inpainted_maps, sparse=True)
        hist.write(output_file, overwrite=True)
        logger.info(f"Inpainted histogram saved to {output_file}")

        return

    def write_csv(self, x_data, y_data, save_prefix, x_label="x", y_label="y"):

        """Save dataframe to file.

        Parameters
        ----------
        x_data : array or list
            Data for first axis.
        y_data : array or list
            Data for seconds axis.
        save_prefix : str
            Prefix of saved file. Will be saved as .dat file.
        x_label : str
            Name of x axis in dat file. Default is 'x'.
        y_label : str
            Name of y axis in dat file. Default is 'y'.
        """

        d = {x_label: x_data, y_label: y_data}
        df = pd.DataFrame(data=d)
        df.to_csv(f"{save_prefix}.dat", float_format='%10.5e', index=False, sep="\t", columns=[x_label, y_label])

        return

    def evaluate_inpainting_accuracy(self, true_bg, estimated_bg, prefix="eval", show_plots=False):

        """Evaluates accuracy of inpainted maps by making series of
        comparison plots, which are saved to file and can also be shown.

        Parameters
        ---------
        true_bg : hist
            Histogram object projected onto Em, Phi, PsiChi with the true background.
        estimated_bg : hist
            Histogram object projected onto Em, Phi, PsiChi with the estimated background.
        prefix : str
            Prefix of saved plots.
        show_plots : bool
            Option to show plots (default is False).
        """

        # Compare projection onto measured energy axis:
        energy = true_bg.axes['Em'].centers
        energy_err = true_bg.axes['Em'].widths / 2.0

        true_plot = true_bg.project('Em').contents.todense()
        est_plot = estimated_bg.project('Em').contents.todense()

        plt.loglog(energy, true_plot, ls="", marker="o", color="darkorange", label="BG true")
        plt.errorbar(energy, true_plot, xerr=energy_err, color="darkorange", ls="", marker="o", label="_nolabel_")

        plt.loglog(energy, est_plot, ls="", marker="s", color="cornflowerblue", label="BG estimated")
        plt.errorbar(energy, est_plot, xerr=energy_err, color="cornflowerblue", ls="", marker="s", label="_nolabel_")

        plt.xlabel("Em [keV]", fontsize=12)
        plt.ylabel("Counts", fontsize=12)
        plt.xlim(1e2, 1e4)
        plt.ylim(1e5, 1e9)
        plt.legend(loc=1, frameon=False)
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_em_counts.png", dpi=150)
        plt.close()

        # Plot fractional difference:
        diff = (est_plot - true_plot) / true_plot

        plt.semilogx(energy, diff, ls="", marker="o", color="black")
        plt.errorbar(energy, diff, xerr=energy_err, ls="", marker="o", color="black")

        plt.xlabel("Em [keV]", fontsize=12)
        plt.ylabel("(estmated - true)/true", fontsize=12)
        plt.xlim(1e2, 1e4)
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_em_frac_diff.png", dpi=150)
        plt.close()

        # save data:
        self.write_csv(energy, diff, "Em_diff", x_label="Em[keV]", y_label="diff")

        # Compare projection onto phi axis:
        phi = true_bg.axes['Phi'].centers
        phi_err = true_bg.axes['Phi'].widths / 2.0

        true_plot = true_bg.project('Phi').contents.todense()
        est_plot = estimated_bg.project('Phi').contents.todense()

        plt.semilogy(phi, true_plot, ls="", marker="o", color="darkorange", label="BG true")
        plt.errorbar(phi, true_plot, xerr=phi_err, color="darkorange", ls="", marker="o", label="_nolabel_")

        plt.semilogy(phi, est_plot, ls="", marker="s", color="cornflowerblue", label="BG estimated")
        plt.errorbar(phi, est_plot, xerr=phi_err, color="cornflowerblue", ls="", marker="s", label="_nolabel_")

        plt.xlabel("Phi [deg]", fontsize=12)
        plt.ylabel("Counts", fontsize=12)
        plt.xlim(0, 200)
        plt.ylim(1e5, 1e9)
        plt.legend(loc=1, frameon=False)
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_phi_counts.png", dpi=150)
        plt.close()

        # Plot fractional difference:
        diff = (est_plot - true_plot) / true_plot

        plt.plot(phi, diff, ls="", marker="o", color="black")
        plt.errorbar(phi, diff, xerr=phi_err, ls="", marker="o", color="black")

        plt.xlabel("Phi [deg]", fontsize=12)
        plt.ylabel("(estmated - true)/true", fontsize=12)
        plt.xlim(0, 200)
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_phi_frac_diff.png", dpi=150)
        plt.close()

        # save data:
        self.write_csv(phi, diff, "phi_diff", x_label="phi[deg]", y_label="diff")

        # Compare projection onto psichi:
        true_plot = true_bg.project('PsiChi')
        true_plot_map = HealpixMap(base=HealpixBase(npix=true_plot.nbins), data=true_plot.contents.todense())
        plot, ax = true_plot_map.plot('mollview')
        plt.title("True BG")
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_psichi_true_bg.pdf")
        plt.close()

        est_plot = estimated_bg.project('PsiChi')
        est_plot_map = HealpixMap(base=HealpixBase(npix=est_plot.nbins), data=est_plot.contents.todense())
        plot, ax = est_plot_map.plot('mollview')
        plt.title("Estimated BG")
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_psichi_estimated_bg.pdf")
        plt.close()

        # Calculate percent diff:
        diff = (est_plot - true_plot) / true_plot
        diff_plot_map = HealpixMap(base=HealpixBase(npix=diff.nbins), data=diff.contents.todense())
        plot, ax = diff_plot_map.plot('mollview', **{"cmap": "bwr", "vmin": -0.3, "vmax": 0.3})
        plt.title("Comparison")
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_psichi_frac_diff.pdf")
        plt.close()

        # save data:
        array = diff.contents.todense()
        self.write_csv(np.arange(array.shape[0]), array, "psichi_diff", x_label="psichi", y_label="diff")

        logger.info(f"Accuracy plots saved with prefix '{prefix}_...'")

        return

    @staticmethod
    def visualize_and_save(input_data_map, mask_map, inpainted_maps,
                           em_bin=2, phi_bin=4, prefix="inpainted", show_plots=False):

        """
        Show and save Mollweide projections of true, masked, and inpainted maps.

        Parameters
        ----------
        input_data_map : array
            Input file with total data. Returned from
            load_projected_data method.
        mask_map : array
            Boolean array specifying mask. Returned from
            mask_from_cumdist_vectorized method.
        inpainted_maps : array
            Estimated background.
        em_bin : int
            Which em bin to use (default is 0).
        phi_bin : int
            Which phi bin to use (default is 0).
        prefix : str
            Prefix of saved files.
        show_plots : bool, optional
            Option to show plots. Default is False.
        """

        maps = [
            (input_data_map[em_bin, phi_bin], f"{prefix}_true_E{em_bin}_Phi{phi_bin}.pdf",
             f"Input Map (Em={em_bin}, Phi={phi_bin})"),
            (input_data_map[em_bin, phi_bin] * mask_map[em_bin, phi_bin],
             f"{prefix}_masked_E{em_bin}_Phi{phi_bin}.pdf", f"Masked Map (Em={em_bin}, Phi={phi_bin})"),
            (inpainted_maps[em_bin, phi_bin], f"{prefix}_inpainted_E{em_bin}_Phi{phi_bin}.pdf",
             f"Inpainted Map (Em={em_bin}, Phi={phi_bin})")]

        for data, filename, title in maps:
            hp.mollview(data, title=title)
            plt.savefig(filename)
            if show_plots == True:
                plt.show()
            plt.close()
            logger.info(f"Saved visualization: {filename}")

        return

    def load_estimated_bg(self, estimated_bg_file):

        """Loads inpainted histrogram from h5 file.

        Parameters
        ----------
        inpainted_file : str
            Path to input h5 file with estimated background.

        """

        self.inpainted_map = Histogram.open(estimated_bg_file)

        return

    def estimate_bg(self, input_data, psr_file, background_model=None,
                    prefix="inpainted", containment=0.6, visualize=False,
                    em_bin=2, phi_bin=4, evaluate_only=False, inpainted_file=None,
                    evaluate=False, show_plots=False, verbose=True):

        """Convenience function for estimating the background.

        Parameters
        ----------
        input_data : str
            Path to HDF5 file with the input data that will be used
            to estimate the background.
        psr_file : str
            Path to point source response HDF5 file.
        background_model : hist, optional
            Optional background model HDF5 file.
        containment : float, optional
            Containment fraction for mask. Default is 0.6.
        prefix : str, optional
            Prefix for output files. Default is 'inpainted'.
        visualize : boolean, optional
            Visualize Mollweide plots.
        em_bin : int, optional
            Energy bin index for visualization. Default is 2.
        phi_bin : int, optional
            Phi bin index for visualization. Default is 4.
        evaluate_only : boolean, optional
            Skip training and evaluate two histograms. Requires
            background_model file and inpainted_file.
        inpainted_file : hist, optional
            Inpainted histogram file (for evaluate-only). Default is None.
        evaluate : boolean
            Evaluate after training (inline). Default is False.
        show_plots : boolean
            Display plots to screen. Default is False.
        verbose : bool, optional
            Gives logger info for validation loss every 50 epochs.
            Default is False.
        """

        # Record run time:
        start_time = time.time()

        # --- Evaluation-only mode (kept compatible) ---
        if evaluate_only is True:
            if not background_model or not inpainted_file:
                raise ValueError("--evaluate-only requires --background_model and --inpainted-file")
            true_hist = Histogram.open(background_model)
            inpainted_hist = Histogram.open(inpainted_file)
            self.evaluate_inpainting_accuracy(true_hist,
                                              inpainted_hist, prefix=prefix, show_plots=show_plots)
            return

        if evaluate and background_model is None:
            raise ValueError("--evaluate requires --background_model")

        # Load data + PSR (same as base class)
        input_data_proj, input_data_map = self.load_projected_data(input_data)
        psr_proj, psr_map = self.load_projected_psr(psr_file)

        # Optional background model (for evaluation)
        model_map = None
        model_proj = None
        if background_model:
            model_proj, model_map = self.load_projected_model(background_model)

        npix = input_data_map.shape[-1]

        # Mask from PSR:
        mask_map = self.mask_from_cumdist_vectorized(psr_map, containment=containment)
        # mask_map is float 0/1; treat "keep" as True
        keep_map = (mask_map == 1)

        # Interpolation inpaint:
        inpainted = np.array(input_data_map, copy=True)

        # Helper: fill a single 1D slice (length npix)
        def _interp_fill_1d(y, keep):
            # y: (npix,), keep: (npix,) bool, where keep==True means NOT masked
            out = y.copy()
            masked_idx = np.where(~keep)[0]
            if masked_idx.size == 0:
                return out

            for i in masked_idx:
                # search left for first unmasked
                left_val = None
                j = (i - 1) % npix
                while j != i:
                    if keep[j]:
                        v = y[j]
                        # accept finite; if non-finite, keep scanning
                        if np.isfinite(v):
                            left_val = v
                            break
                    j = (j - 1) % npix

                # search right for first unmasked AND non-zero
                right_val = None
                j = (i + 1) % npix
                while j != i:
                    if keep[j]:
                        v = y[j]
                        if np.isfinite(v):
                            right_val = v
                            break
                    j = (j + 1) % npix

                # assign with edge-case handling
                if (left_val is not None) and (right_val is not None):
                    out[i] = 0.5 * (left_val + right_val)
                elif left_val is not None:
                    out[i] = left_val
                elif right_val is not None:
                    out[i] = right_val
                else:
                    out[i] = 0.0

            return out

        # Loop over (Em, Phi) planes
        # input_data_map is dense array from Histogram.project([...]).contents.todense()
        # shape expected: (nEm, nPhi, npix)
        nEm = inpainted.shape[0]
        nPhi = inpainted.shape[1]
        for e in range(nEm):
            for p in range(nPhi):
                inpainted[e, p] = _interp_fill_1d(inpainted[e, p], keep_map[e, p])

        self.inpainted_maps = inpainted

        # Save result
        self.save_inpainted_histpy(input_data_proj, self.inpainted_maps, output_file=f"{prefix}_estimated_bg.h5")

        #  Visualization:
        if visualize:
            self.visualize_and_save(input_data_map, mask_map, self.inpainted_maps,
                                    em_bin=em_bin, phi_bin=phi_bin, prefix=prefix, show_plots=show_plots)

        # Inline evaluation:
        if evaluate:
            inpainted_hist = Histogram(input_data_proj.axes, contents=self.inpainted_maps, sparse=True)
            self.evaluate_inpainting_accuracy(model_proj, inpainted_hist, prefix=prefix, show_plots=show_plots)

        end_time = time.time()
        logger.info(f"Total time elapsed: {end_time - start_time:.2f} seconds")

        return
