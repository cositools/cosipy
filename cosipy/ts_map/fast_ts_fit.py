from pathlib import Path

from enum import Enum

import numpy as np
import numba

import matplotlib.pyplot as plt

import healpy as hp
from mhealpy import HealpixBase

from cosipy import SpacecraftFile
from cosipy.response import FullDetectorResponse, GalacticResponse
from cosipy.response.functions import get_integrated_spectral_model

from .fast_norm_fit import FastNormFit as fnf

import logging
logger = logging.getLogger(__name__)


class Frame(Enum):
    LOCAL = 1
    GALACTIC = 2

class FastTSMap():

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

        match cds_frame:
            case "galactic":
                self._cds_frame = Frame.GALACTIC
            case "local":
                self._cds_frame = Frame.LOCAL
            case _:
                raise TypeError(f"Unrecognized frame {cds_frame}, "
                                "must be 'local' or 'galactic'")

        if self._cds_frame == Frame.LOCAL:
            if orientation is None:
                raise TypeError("When data are binned in local frame, "
                                "orientation must be provided")

            self._orientation = orientation

            self._response = FullDetectorResponse.open(response_path)
        else:

            self._response = GalacticResponse.open(response_path)

        labels = self._response.axes.labels

        # mapping only works with CDS's consisting of Em/Phi/PsiChi
        # (in any order). The response must map from NuLambda / Ei to
        # the CDS.

        if not all(labels[:2] == ("NuLambda", "Ei")):
            raise ValueError("Response axes must begin with (NuLambda, Ei)")

        # extract order of response's CDS dimensions for linearization
        # of data, bkg

        cds_order = tuple(labels[2:])
        if not all(ax in ("Em", "Phi", "PsiChi") for ax in cds_order):
            raise ValueError("Response CDS axes must be Em/Phi/PsiChi")

        # make sure data and background CDS are ordered to match response
        self._data = data.todense().project(cds_order)
        self._bkg_model = bkg_model.todense().project(cds_order)

        self._fnf = fnf(max_iter=1000)

    @staticmethod
    def _get_hypothesis_coords(nside, pixels = None,
                               scheme = "nested",
                               coordsys = "galactic"):
        """
        Get directions corresponding to pixels of a HEALPix map of a
        given resolution and scheme.

        Parameters
        ----------
        nside : int
            Nside of HEALPix map
        pixels : array-like of int, optional
            Array of pixels to convert to directions; if not
            specified, directions will be generated for every pixel in
            map
        scheme : str, optional
            Scheme of HEALPix map ("ring" or "nested"; default: nested)
        coordsys : str, optional
            Coordinate system of HEALPix map (default: galactic)

        Returns
        -------
        hypothesis_coords : np.ndarray of (# pixels x 3)
            Cartesian 3-vectors for each pixel's direction

        """

        if pixels is None:
            npix = hp.nside2npix(nside)
            pixels = np.arange(npix, dtype=int)

        hpbase = HealpixBase(nside = nside, scheme = scheme,
                             coordsys = coordsys)

        return np.column_stack(hpbase.pix2vec(pixels))

    @staticmethod
    def _get_cds_array(hist, em_slice):
        """
        Convert a CDS histogram to a flattened array, projecting over
        just the selected channels of the Em dimension.

        Parameters
        -----------
        hist : histpy.Histogram
           A CDS count Histogram
        em_slice : Slice object
           Energy (Em) channels to use in fitting

        Returns
        -------
        cds_array : numpy.ndarray
            Flattened CDS array

        """

        hist_cds_sliced = hist.slice[{"Em" : em_slice}]
        hist_cds = hist_cds_sliced.project_out("Em")

        cds_array = hist_cds.contents
        if hist_cds.unit is not None:
            cds_array = cds_array.value

        return cds_array.ravel()

    def _fit_one_direction(self, source,
                           data_cds_array, bkg_model_cds_array,
                           psr_cache):
        """
        Perform a TS fit of data for a single source direction

        Parameters
        ----------
        source : np.ndarray
            source direction as Cartesian 3-vector
        data_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the data.
        bkg_model_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the
            background model.
        psr_cache : PSRCache
            Cache to retrieve PSR for source direction

        Returns
        -------
        result of TS fitting:
          [ts value, norm, norm_err, failed, # iterations]

        """

        if self._cds_frame == Frame.LOCAL:

            # convert source direction to path in local frame
            lons, colats = self._orientation.get_target_in_sc_frame(source)

            # get list of HEALPix pixels with nonzero exposure on path
            pixels, exposures = \
                self._orientation.get_exposure(base = self._response,
                                               theta = colats,
                                               phi = lons,
                                               lonlat = False)
        else: # galactic frame

            # convert source vector to polar coords
            x, y, z = source
            lon   = np.arctan2(y, x)
            colat = np.arccos(z)

            # interpolate the source onto the response grid
            pixels, exposures = \
                self._response.get_interp_weights(theta = colat,
                                                  phi = lon,
                                                  lonlat = False)

        # sum the PSRs for each NuLambda pixel according to their
        # exposure weights
        ei_cds_array = np.zeros(psr_cache.shape, psr_cache.dtype)
        ei_sum = 0.

        for p, exposure in zip(pixels, exposures):
            psr, psr_sum = psr_cache.get_psr(p)
            ei_cds_array += psr * exposure
            ei_sum += psr_sum * exposure

        return self._fnf.solve(data_cds_array, bkg_model_cds_array,
                               ei_cds_array, ei_sum)

    def _prepare_inputs(self, energy_channel, spectrum, max_cache_size):
        """
        Prepare the data and background arrays for ts fitting, and get ready
        to read and cache PSRs for different source directions.  The shape
        and contents of the arrays and PSRs depends on the data reductions
        implied by the energy channel and spectrum.

        Parameters
        ----------
        energy_channel : 2-element list [lower_channel, upper_channel]
            Energy (Em) channels to use in fitting (Python range
            lower_channel:upper_channel)
        spectrum : astromodels.functions
            Spectrum of the source.
        max_cache_size : int or None
            Maximum number of entries to store in PSRCache (None = no limit)

        Returns
        -------
        data_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the data.
        bkg_model_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the
            background model.
        psr_cache : PSRCache
            Cache to retrieve PSR for source directions

        """

        if energy_channel is None:
            em_slice = slice(None)
        else:
            em_slice = slice(energy_channel[0], energy_channel[1])

        # get the flattened data and background CDS arrays
        data_cds_array = self._get_cds_array(self._data, em_slice)
        bkg_model_cds_array = self._get_cds_array(self._bkg_model, em_slice)

        # eliminate CDS cells with no counts in data (due to data
        # sparsity) or in bkg model (lack of pseudocounts in bkg model
        # -- could be considered a bug, may cause divide-by-zero error
        # in fitting)
        valid_cells = np.where(np.logical_and(data_cds_array > 0,
                                              bkg_model_cds_array > 0))[0]

        data_cds_array = data_cds_array[valid_cells]
        bkg_model_cds_array = bkg_model_cds_array[valid_cells]

        flux = get_integrated_spectral_model(spectrum, self._response.axes["Ei"])

        psr_cache = PSRCache(self._response, em_slice, valid_cells, flux,
                             maxSize = max_cache_size)

        return data_cds_array, bkg_model_cds_array, psr_cache

    def fit(self, nside, spectrum, energy_channel = None,
            cpu_cores = None, max_cache_size = None):
        """
        Produce a ts map of specified resolution.

        Parameters
        ----------
        nside : int
            HEALPix nside of ts map to produce
        spectrum : astromodels.functions
            Spectrum of the source.
        energy_channel : 2-element list, of form
                         [lower_channel, upper_channel], optional
            Energy (Em) channels to use in fitting (Python range
            lower_channel:upper_channel). If not specified, use all
            Em channels.
        cpu_cores : int, optional
            Number of processors to use (default: do not restrict)
        max_cache_size : int, optional
            Maximum number of entries to store in PSRCache; if None,
            no limit

        Returns
        -------
        results : numpy.ndarray
            Fitted ts values for each hypothesis coordinate

        """

        if cpu_cores is not None:
            numba.set_num_threads(cpu_cores)

        data_cds_array, bkg_model_cds_array, psr_cache = \
            self._prepare_inputs(energy_channel, spectrum, max_cache_size)

        if self._cds_frame == Frame.LOCAL:
            # compute possible source dirs in same frame
            # we will use to translate them to local-frame paths
            hyp_frame = self._orientation.frame
        else: # galactic frame
            hyp_frame = "galactic"

        hypothesis_coords = self._get_hypothesis_coords(nside,
                                                        coordsys=hyp_frame)

        results = [
            self._fit_one_direction(source,
                                    data_cds_array,
                                    bkg_model_cds_array,
                                    psr_cache)[0]
            for source in hypothesis_coords
        ]

        return np.array(results)

    @staticmethod
    def plot_ts(m_ts, skycoord = None, containment = None, scheme="nested",
                save_plot = False, save_dir = "",
                save_name = "ts_map.png", dpi = 300):
        """
        Plot a TS map.

        Parameters
        ----------
        m_ts : numpy.ndarray
            The array of ts values from a ts fit.
        skycoord : astropy.coordinates.SkyCoord, optional
            The true location of the source (default: do not plot)
        containment : float, optional
            Restrict the plotted pixels to the specified containment
            threshold relative to the max ts value (default: plot
            *all* ts values)
        scheme : string, optional
            HEALPix scheme of ts map values ("ring" or "nested";
            default = "nested")
        save_plot : bool, optional
            Save the plot to a file (default: False)
        save_dir : string, optional
            Directory in which to save the plot
        save_name : str, optional
            File name under which tos ave the plot
        dpi : int, optional
            DPI used for plotting / saving

        """

        fig, ax = plt.subplots(dpi = dpi)
        nest = scheme.startswith("nest")

        if containment is not None:
            critical = FastTSMap.get_chi_critical_value(containment = containment)
            max_ts = np.max(m_ts)
            hp.mollview(m_ts, max = max_ts, min = max_ts - critical,
                        nest=nest,
                        title = f"Containment {containment*100}%",
                        coord = "G",
                        hold = True)
        else:
            hp.mollview(m_ts, nest=nest, coord = "G", hold = True)

        if skycoord is not None:
            lon = skycoord.l.deg
            lat = skycoord.b.deg
            hp.projscatter(lon, lat, marker = "x",
                           linewidths = 0.5,
                           lonlat=True,
                           coord = "G",
                           label = f"True location at l={lon}, b={lat}",
                           color = "fuchsia")

        hp.projscatter(0, 0, marker = "o",
                       linewidths = 0.5,
                       lonlat=True,
                       coord = "G",
                       color = "red")

        hp.projtext(350, 0, "(l=0, b=0)",
                    lonlat=True,
                    coord = "G",
                    color = "red")

        if save_plot:
            fig.savefig(Path(save_dir)/save_name, dpi = dpi)

        plt.show()
        plt.close(fig)

    @staticmethod
    def get_chi_critical_value(containment = 0.90):
        """
        Get the critical value of the chi^2 distribution based on the
        confidence level.

        Parameters
        ----------
        containment : float, optional
          The confidence level of the chi^2 distribution (the default is
          `0.9`, which implies that the 90% containment region).

        Returns
        -------
        float
            The critical value corresponding to the confidence level.

        """

        from scipy.stats import chi2

        return chi2.ppf(containment, df=2)


class PSRCache:
    """
    A cached reader for PSR data from a response file, designed for
    use with FastTSMap.  For a given NuLambda pixel p, we fetch the
    pixel's data from the underlying response file and do all the data
    reduction needed to compute a PSR for pixel p averaged over the
    input flux.  The result is cached so that, when different source
    directions require a PSR for the same pixel p, we don't do the
    fetching and reduction more than once.

    If memory usage is a concern, the cache can be set to a given max
    size with LRU replacement.  But the reduced PSR sums away the Ei
    and Em dimensions *and* removes CDS voxels that do not matter for
    the ts_map computation, so it is much smaller than a raw chunk of
    the response file. Hence, it is likely not necessary to limit the
    cache size in practice.

    """

    def __init__(self, response, em_slice, valid_cells, flux,
                 maxSize = None):
        """
        Create a new PSRCache, providing the information needed
        to fetch and reduce PSRs from the response file on demand.

        Parameters
        ----------
        response : FullDetectorResponse
          The response from which to read slices for PSR computation
        em_slice : Slice object
          The slice of the Em axis used to compute PSRs
        valid_cells : np.ndarray of int
          CDS voxels on the linearized Phi/PsiChi axis that are actually
          used in in the ts_map computation
        flux : Histogram
          Integrated spectral flux, binned according to response's Ei axis
        maxSize: int (optional)
          If not None, maximum number of NuLambda pixels for which we will
          cache PSRs.  The cache is managed according to an LRU policy.

        """

        from collections import OrderedDict

        self.cache = OrderedDict()
        self.maxSize = maxSize

        self.response = response
        self.em_axis = response.axes.label_to_index("Em") - 1 # for NuLambda
        self.em_slice = em_slice
        self.valid_cells = valid_cells

        self.ei_weights = flux.contents.value * response.eff_area_correction

        #self.nLookups = 0
        #self.nMisses = 0

    @property
    def shape(self):
        """
        Array shape of a PSR returned by the cache
        """
        return (len(self.valid_cells),)

    @property
    def dtype(self):
        """
        Element type of a PSR returned by the cache
        """
        return self.response.dtype

    def get_psr(self, p):
        """
        Get the reduced PSR for NuLambda pixel p.

        Parameters
        ----------
        p : int
          NuLambda value of requested PSR

        Returns
        -------
        psr : np.ndarray of float (length = |valid_cells|)
          PSR for pixel p, summed over requested Em and
          convolved with spectral flux.  The result gives
          one value per valid voxel.
        psr_sum
          Sum of PSR for pixel p over *all* voxels, not just
          the valid ones.

        """

        #self.nLookups += 1
        v = self.cache.get(p)
        if v is None: # cache miss
            #self.nMisses += 1
            v = self._compute_psr(p)
            self.cache[p] = v

            # implement LRU policy if requested
            if self.maxSize is not None and len(self.cache) > self.maxSize:
                self.cache.popitem(last=False)
        else:
            # move MRU value to end to support LRU policy if requested
            if self.maxSize is not None:
                self.cache.move_to_end(p)

        return v

    '''
    def print_stats(self):
        """
        Print cache miss statistics
        """

        missRate = 0. if self.nLookups == 0 else self.nMisses/self.nLookups

        print(f"Cache size: {len(self.cache)} (out of {self.maxSize})")
        print(f"Misses: {self.nMisses} / {self.nLookups} = {missRate:0.3f}")
    '''

    def _compute_psr(self, p):
        """
        Compute a reduced PSR for NuLambda pixel p.

        Returns
        -------
        psr : np.ndarray of float (length = |valid_cells|)
          PSR for pixel p, summed over requested Em and
          convolved with spectral flux.  The result gives
          one value per valid voxel.
        psr_sum
          Sum of PSR for pixel p over *all* voxels, not just
          the valid ones.

        """

        # get raw CDS counts for pixel, trimmed by Em slice size is Ei
        # x (Em, Phi, PsiChi) in some order
        counts = self.response.get_counts(p, self.em_slice)

        # sum over Em dimension and convert to float : Ei x Phi/PsiChi
        counts = np.sum(counts, axis=self.em_axis, dtype=self.response.dtype)

        # linearize CDS : Ei x CDS voxels. Note that we ensure in
        # FastTSMap that data and bkg will use the same dimension
        # ordering as the response for the CDS, so there is no need to
        # re-order dimensions here.
        counts = counts.reshape(counts.shape[0], -1)

        # extract valid CDS voxels of psr after capturing sum of *all*
        # voxels : Ei x valid CDS voxels
        psr_sum = np.sum(counts, axis=1)
        psr = counts[:, self.valid_cells]

        # convolve psr with flux (and also eff_area correction
        # weights, which have not yet been applied) to remove Ei
        # dimension
        psr_sum = np.dot(psr_sum, self.ei_weights)
        psr = np.tensordot(psr, self.ei_weights, axes=(0,0))

        return psr, psr_sum
