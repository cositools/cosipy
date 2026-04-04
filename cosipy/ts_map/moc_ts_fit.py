from pathlib import Path

import numpy as np
import numba

import mhealpy as hp

import matplotlib.pyplot as plt

from .fast_ts_fit import FastTSMap, Frame

import logging
logger = logging.getLogger(__name__)

class MOCTSMap(FastTSMap):
    """
    Multi-resolution source mapping.
    """

    def __init__(self, data, bkg_model, response_path, orientation = None,
                 cds_frame = "local"):
        """
        Initialize the instance of a TS map fit.

        Parameters
        ----------
        data : histpy.Histogram
            Observed data, which includes counts from both signal and
            background.
        bkg_model : histpy.Histogram
            Model used to estimate background counts in observed data.
        response_path : str or pathlib.Path
            Path to response file.
        orientation : cosipy.SpacecraftFile, optional
            Orientation history of spacecraft; required for "local"
            cds_frame, not used if frame is "galactic"
        cds_frame : str, optional
            frame of directions used for PsiChi axis of CDS.  One of
            "local" (frame attached to spacecraft) or "galactic".
            Default is local.

        """

        super().__init__(data, bkg_model, response_path,
                         orientation = orientation,
                         cds_frame = cds_frame)

    class Strategy:
        """
        A generic strategy API for selecting pixels to refine in a moc map.
        """

        def __call__(self, ts, pixels, nside):
            """
            Select a subset of input pixels to refine.

            Parameters
            ----------
            ts : np.array of float
               ts values for a set of pixels
            pixels : np.array of int
               nested-scheme indices for pixels corresponding to each ts
               value
            nside : int
               nside of map from which pixels are drawn

            Returns
            -------
              boolean mask -- True only for those pixels that should be
              refined

            """
            raise RuntimeError("Strategy subclass must redefine select()")

    class TopKStrategy(Strategy):
        """
        Refine a fixed number of pixels with the highest ts values
        """

        def __init__(self, k):
            """
            Parameters
            ----------
            k : int
              refine the k pixels with highest ts values
            """

            self.k = k

        def __call__(self, ts, pixels, nside):
            top_ts_indices = np.argpartition(ts, -self.k)[-self.k:]
            hi_idx = top_ts_indices[-self.k:]
            hi_mask = np.zeros(len(ts), dtype=bool)
            hi_mask[hi_idx] = True

            return hi_mask

    class ContainmentStrategy(Strategy):
        """
        Refine all pixels within a specified containment region
        based on ts value
        """

        def __init__(self, containment):
            """
            Parameters
            ----------
            containment : float
              refine pixels whose ts value is within the specified
              containment region vs the max value.
            """

            self.chi = FastTSMap.get_chi_critical_value(containment)

        def __call__(self, ts, pixels, nside):
            return (ts >= ts.max() - self.chi)

    class PaddingStrategy(Strategy):
        """
        After applying a specified strategy, pad the result to include
        all neighbors of pixels chosen to be refined.
        """

        def __init__(self, sub_strategy):
            """
            Parameters
            ----------
            sub_strategy : MOCTSMap.Strategy subclass
              strategy to apply prior to padding
            """

            self.sub_strategy = sub_strategy

        def __call__(self, ts, pixels, nside):

            hi_mask = self.sub_strategy(ts, pixels, nside)

            hi_adj = hp.get_all_neighbours(nside, pixels[hi_mask], nest=True)
            hi_adj = np.unique(hi_adj)
            adj_mask = np.isin(pixels, hi_adj, assume_unique=True)
            hi_mask[adj_mask] = True

            return hi_mask

    def fit(self, max_nside, spectrum, energy_channel = None,
            cpu_cores = None, max_cache_size = None,
            init_nside = 1, strategy = None):
        """
        Construct a multi-resolution map of ts statistics, selectively
        refining the highest-scoring pixels.

        Parameters
        ----------
        max_nside : int
          highest possible nside reached during refinement
        spectrum : astromodels.functions
          spectrum of the source.
        energy_channel : 2-element list, of form
                         [lower_channel, upper_channel], optional
            Energy (Em) channels to use in fitting (Python range
            lower_channel:upper_channel). If not specified, use all
            Em channels.
        cpu_cores : int, optional
          number of processors to use (default: do not restrict)
        max_cache_size : int, optional
            Maximum number of entries to store in PSRCache; if None,
            no limit
        init_nside : int, optional
          lowest nside used in map
        strategy : MOCTSMap.Strategy subclass, optional
          strategy to use in selecting pixels to refine.  If None,
          default to PaddingStrategy(ContainmentStrategy(0.999)).

        Returns
        -------
        ts : np.ndarray
          ts statistics for each pixel in map
        uniqs : np.ndarray of int
          uniq pixel IDs for each output pixel in map

        """

        def refine(pix):
            """
            Given pixels in NEST format, expand each to
            its four sub-pixels at the next nside.
            """

            res = np.tile(pix, 4)
            res *= 4

            n = len(pix)
            res[n:2*n]   += 1
            res[2*n:3*n] += 2
            res[3*n:]    += 3

            return res

        if strategy is None:
            self.strategy = self.PaddingStrategy(self.ContainmentStrategy(0.999))
        else:
            self.strategy = strategy

        if cpu_cores is not None:
            numba.set_num_threads(cpu_cores)

        data_cds_array, bkg_model_cds_array, psr_cache = \
            self._prepare_inputs(energy_channel, spectrum, max_cache_size)

        all_pix = []
        all_ts = []

        # initially, compute ts for all pixels at minimum nside
        nside = init_nside
        pixels = np.arange(hp.nside2npix(init_nside), dtype=int)

        while nside <= max_nside:

            if self._cds_frame == Frame.LOCAL:
                # compute possible source dirs in same frame
                # we will use to translate them to local-frame paths
                hyp_frame = self._orientation.attitude.frame
            else: # galactic frame
                hyp_frame = "galactic"

            src_locs = self._get_hypothesis_coords(nside, pixels,
                                                   coordsys=hyp_frame)

            results = [
                self._fit_one_direction(source,
                                        data_cds_array,
                                        bkg_model_cds_array,
                                        psr_cache)[0]
                for source in src_locs
            ]

            ts = np.array(results)

            if nside == max_nside:
                # Done -- save all remaining pixels and their values
                all_pix.append(hp.nest2uniq(nside, pixels))
                all_ts.append(ts)
                break

            hi_mask = self.strategy(ts, pixels, nside)

            # For pixels that we will *not* refine, compute their
            # unique indices and save them.
            lo_mask = ~hi_mask
            lo_pix = hp.nest2uniq(nside, pixels[lo_mask])
            lo_ts  = ts[lo_mask]

            all_pix.append(lo_pix)
            all_ts.append(lo_ts)

            # Split pixels that we *will* refine down to next nside
            pixels = refine(pixels[hi_mask])

            nside *= 2

        return np.concatenate(all_ts), np.concatenate(all_pix)

    @staticmethod
    def plot_ts(moc_ts, moc_uniq,
                skycoord = None, containment = None,
                grid_lines = True,
                save_plot = False, save_dir = "",
                save_name = "ts_map.png", dpi = 300):
        """
        Plot a multi-resolution TS map.

        Parameters
        ----------
        moc_ts : np.ndarray
            ts values of multiresolution map
        moc_uniq : np.ndarray of int
            uniq HEALPixel values of multiresolution map
        skycoord : astropy.coordinates.SkyCoord, optional
            The true location of the source (default: do not plot)
        containment : float, optional
            Restrict the plotted pixels to the specified containment
            threshold relative to the max ts value (default: plot
            *all* ts values)
        grid_lines : bool, optional
            Print lines bordering each pixel in the map (default: True)
        save_plot : bool, optional
            Save the plot to a file (default: False)
        save_dir : string, optional
            Directory in which to save the plot
        save_name : str, optional
            File name under which tos ave the plot
        dpi : int, optional
            DPI used for plotting / saving

        """

        moc_map = hp.HealpixMap(data = moc_ts, uniq = moc_uniq)

        # get plotting canvas
        fig = plt.figure(dpi = dpi)

        axMoll = fig.add_subplot(1,1,1, projection = 'mollview')

        if containment is None:
            moc_map.plot(ax = axMoll)
        else:
            critical = FastTSMap.get_chi_critical_value(containment = containment)
            max_ts = np.max(moc_ts)
            moc_map.plot(ax = axMoll,
                         vmin = max_ts - critical,
                         vmax = max_ts)

        if grid_lines:
            moc_map.plot_grid(ax = plt.gca(), color = 'grey',
                              linewidth = 0.1);

        # plot the source location if given
        if skycoord is not None:

            axMoll.text(skycoord.l.deg, skycoord.b.deg, "x", size = 4,
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = axMoll.get_transform('world'),
                        color = "red")

        if save_plot:
            fig.savefig(Path(save_dir)/save_name, dpi = dpi)

        plt.show()
        plt.close(fig)
