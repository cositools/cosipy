import numpy as np

import matplotlib.pyplot as plt

from mhealpy import HealpixMap

from cosipy import BinnedData

import logging
logger = logging.getLogger(__name__)

class ContinuumEstimation:

    def mask_from_cumdist(self, psichi_vals, containment, make_plots=False):

        """Determines masked pixels from cumulative distribution of the point
        source response.

        Parameters
        ----------
        psichi_vals : array
            Values of point source response projected onto psichi axis.
        containment : float
            The percentage (non-inclusive) of the cumulative
            distribution to use for the mask, i.e. all pixels that
            fall below this value in the cumulative distribution will
            be masked.
        make_plots : bool, optional
            Option to plot cumulative distribution.

        Returns
        -------
        masked_indices : array of int
            Indices in map that should be masked

        Note
        ----
        The cumulative distribution is an estimate of the angular
        resolution measure (ARM), which is a measure of the PSF for
        Compton imaging.

        """

        v_sum = np.sum(psichi_vals)

        # get indices needed to sort data in descending order
        sorted_indices = np.argsort(psichi_vals)[::-1]
        v_sorted = psichi_vals[sorted_indices]

        # Define mask based on fraction of total exposure
        # (i.e. counts)
        v_cum = np.cumsum(v_sorted)
        arm_mask = (v_cum < containment * v_sum)

        # Plot cumulative distribution and corresponding masks
        if make_plots:
            plt.plot(v_cum / v_sum)
            plt.title("Cumulative Distribution")
            plt.xlabel("Pixel")
            plt.ylabel("Fraction of Counts")
            plt.show()

        return sorted_indices[arm_mask]

    def simple_inpainting(self, m_data, mask_indices):

        """Highly simplistic method for inpainting masked region in CDS.

        This method relies on the input healpix map having a ring
        ordering. For each masked pixel, it searches to the left (i.e.
        lower pixel numbers) until reaching the first non-zero pixel.
        It then search to the right (i.e. higher pixel numbers) until
        again finding the first non-zero pixel. The mean of the two
        values is used for filling in the masked pixel.

        Parameters
        ----------
        m_data : array-like
            HealpixMap object, containing projection of PSR onto
            psichi.
        mask_indices : array of int
            Indices in m_data that have been masked.

        Returns
        -------
        array
            Values for the inpainting, corresponding to the masked
            pixels.

        """

        # Get mean of masked data for edge cases (simple solution for now):
        # CK: It would be better if this were at least the mean of an
        # np masked array object, but a better method is anyways needed.
        masked_mean = np.mean(m_data)

        # indices of all nonzero entries in m_data
        nz_indices = np.nonzero(m_data)[0]

        # for each index in mask_indices, find index of next highest
        # *nonzero* entry in m_data. (This is always strictly higher
        # than the mask index, which is zero in m_data.)
        next_nonzero = np.searchsorted(nz_indices, mask_indices)

        # get value at last nonzero index of m_data to left of each
        # mask index.  If no nonzero index to left, fill with
        # masked_mean.
        lo_vals = m_data[nz_indices[np.maximum(next_nonzero - 1, 0)]]
        lo_vals[next_nonzero == 0] = masked_mean

        # get value at first nonzero index of m_data to right of each
        # mask index.  If no nonzero index to right, fill with
        # masked_mean.
        maxval = len(nz_indices)
        hi_vals = m_data[nz_indices[np.minimum(next_nonzero, maxval - 1)]]
        hi_vals[next_nonzero == maxval] = masked_mean

        return 0.5 * (lo_vals + hi_vals)


    def continuum_bg_estimation(self, data_file, data_yaml, psr,
                                containment=0.4, make_plots=False,
                                e_loop=None, s_loop=None):

        """Estimates continuum background.

        Parameters
        ----------
        data_file : str
            Full path to binned data (must be .h5 file).
        data_yaml : str
            Full path to the dataIO yaml file used for binning the data.
        psr : py:class:`PointSourceResponse`
            Point source response object.
        containment : float, optional
            The percentage (non-inclusive) of the cumulative distribution
            to use for the mask, i.e. all pixels that fall below this value
            in the cumulative distribution will be masked. Default is 0.4.
        make_plots : bool, optional
            Option to make some plots of the data, response, and masks.
            Default is False.
        e_loop : tuple, optional
            Option to pass tuple specifying which energy range to
            loop over. This must coincide with the energy bins. The default
            is all bins.
        s_loop : tuple, optional
            Option to pass tuple specifying which Phi anlge range to
            loop over. This must coincide with the Phi  bins. The default
            is all bins.

        Returns
        -------
        estimated_bg : histpy:Histogram
            Estimated background as histpy object.
        """

        # Load data to be used for BG estimation
        full_data = BinnedData(data_yaml)
        full_data.load_binned_data_from_hdf5(data_file)

        # extract just the CDS projection of the BG data
        estimated_bg = \
            full_data.binned_data.project('Em', 'Phi', 'PsiChi').todense()

        # project the psr to just the CDS coordinates
        psr_data = psr.project('Em', 'Phi', 'PsiChi').contents

        # Defaults for energy and scattering angle loops
        if e_loop is None:
            e_loop = (0, psr.axes['Em'].nbins)
        if s_loop is None:
            s_loop = (0, psr.axes['Phi'].nbins)

        # Loop through all bins of energy and phi
        for E in range(*e_loop):
            for s in range(*s_loop):

                # Get mask from PSR for current E, s
                mask_indices = self.mask_from_cumdist(psr_data[E, s],
                                                      containment,
                                                      make_plots=make_plots)

                if len(mask_indices) == 0:
                    continue

                if make_plots:
                    # Plot original response
                    m_dummy = HealpixMap(base = psr.axes["PsiChi"],
                                         data = psr_data[E, s])
                    plot,ax = m_dummy.plot('mollview')
                    plt.title("Interpolated Data (Estimated BG)")
                    plt.show()

                    # Plot masked response
                    m_dummy[mask_indices] = 0 * m_dummy.unit
                    plot,ax = m_dummy.plot('mollview')
                    plt.title("Masked Response")
                    plt.show()

                # get BG data for current E, s; we will modify
                # the estimated_background Histogram's contents
                # in place.
                m_data = estimated_bg.contents[E, s]

                if make_plots:
                    # Plot unmasked data
                    m_dummy = HealpixMap(base = psr.axes["PsiChi"],
                                         data = m_data)
                    plot,ax = m_dummy.plot('mollview')
                    plt.title("Interpolated Data (Estimated BG)")
                    plt.show()

                # apply the mask
                m_data[mask_indices] = 0

                if make_plots:
                    # Plot masked data
                    m_dummy = HealpixMap(base = psr.axes["PsiChi"],
                                         data = m_data)
                    plot,ax = m_dummy.plot('mollview')
                    plt.title("Interpolated Data (Estimated BG)")
                    plt.show()

                # Get interpolated values
                interp_vals = self.simple_inpainting(m_data, mask_indices)

                # replace masked pixels with interpolated values
                m_data[mask_indices] = interp_vals

                if make_plots:
                    # Plot masked data with interpolated values
                    m_dummy = HealpixMap(base = psr.axes["PsiChi"],
                                         data = m_data)
                    plot,ax = m_dummy.plot('mollview')
                    plt.title("Interpolated Data (Estimated BG)")
                    plt.show()
                    plt.close()

        return estimated_bg
