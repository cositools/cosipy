import numpy as np

from numba import jit, prange

import logging
logger = logging.getLogger(__name__)

class FastNormFit:
    """Perform a fast poisson maximum likelihood ratio fit of the
    normalization of a source over background.

    The likelihood ratio as a function of the norm is computed as
    follow

    .. math::

        TS(N) = 2 \\sum_i \\left( \\frac{\\log P(d_i; b_i+N e_i)}{\\log P(d_i; b_i)} \\right)

    where :math:`P(d; \\lambda)` is the Poisson probability of
    obtaining :math:`d` count where :math:`\\lambda` is expected on
    average; :math:`b` is the estimated number of background counts;
    :math:`N` is the normalization; and :math:`e` is the expected
    excess -i.e. signal- per normalization unit -i.e. the number of
    excess counts equals :math:`N`.

    It can be shown that :math:`TS(N)` has analytic derivative of
    arbitrary order and that the Newton's method is guaranteed to
    converge if initialized at :math:`N=0`.

    .. note::

        The background is not allowed to float. It is assumed the
        error on the estimation of the background is small compared to
        the fluctuation of the background itself
        (i.e. :math:`N_{B}/N_{off} \\lll 1`).

    .. note::

        Because of the Poisson probability, :math:`TS(N)` is only
        well-defined for :math:`N \\geq 1`. By default,
        underfluctuations are set to :math:`TS(N=0) = 0`. For cases
        when there is benefit in letting the normalization float to
        negative values, you can use `allow_negative`, but in that
        case the results are only valid in the Gaussian regime.

    Args:
        max_iter (int):
           Maximum number of iteration
        conv_frac_tol (float):
           Convergence stopping condition, expressed as the ratio
           between the size of the last step and the current norm
           value.
        zero_ts_tol (float):
            If zero_ts_tol < TS < 0, then TS is set to 0 without
            failed flag status (analytically, TS < 0 should never
            happen)
        allow_negative (bool):
            Allow the normalization to float toward negative values

    """

    def __init__(self,
                 max_iter = 1000,
                 conv_frac_tol = 1e-3,
                 zero_ts_tol = 1e-5,
                 allow_negative = False):

        self.max_iter = max_iter
        self.conv_frac_tol = conv_frac_tol
        self.zero_ts_tol = zero_ts_tol
        self.allow_negative = allow_negative

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def ts(data, bkg, unit_excess, ue_sum, norm):
        """
        Get TS for a given normalization.

        Args:
            data (array): Observed counts
            bkg (array): Background estimation. Same size as data.
                Every element should be >0
            unit_excess (array): Expected excess per normalization unit.
                Same size as data.
            norm (float or array): Normalization value

        Return:
            float: TS
        """

        ts = - norm * ue_sum

        # NB: on Intel processors, np.log1p, like most
        # transcendentals, uses a much slower implementation in Numba
        # than in regular Numpy. Install the icc_rt package and make
        # sure Numba has SVML support enabled (run numba -s to check)
        # to fix.
        for i in prange(len(data)):
            ts += data[i] * np.log1p(norm * unit_excess[i]/bkg[i])

        return 2 * ts

    '''
    @staticmethod
    def ts_serial(data, bkg, unit_excess, ue_sum, norm):
        """
        Non-Numba impl of ts()
        """

        ts = np.sum(data * np.log1p(norm * unit_excess/bkg))

        return 2 * (ts - norm * ue_sum)
    '''

    '''
    @staticmethod
    def dts(data, bkg, unit_excess, ue_sum, norm, order=1):
        """
        Get the derivative of TS with respect to the normalization,
        at given normalization.  (Note: this function is replaced
        with various accelerated versions in solve() below.)

        Args:
            data (array): Observed counts
            bkg (array): Background estimation. Same size as data.
                Every element should be >0
            unit_excess (array): Expected excess per normalization unit.
                Same size as data.
            norm (float or array): Normalization value
            order (int): Derivative order

        Return:
            float or array: d^n TS / dN^n, same shape as norm
        """
        from math import factorial

        assert order >= 1

        r = unit_excess / (bkg + norm * unit_excess)

        dts = np.sum(data * r**order)

        if order == 1:
            dts -= ue_sum

        dts *= 2 * ((-1)**(order-1)) * factorial(order-1)

        return dts
    '''

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def d1ts(data, bkg, unit_excess, ue_sum, norm):
        """
        Get the first derivative of TS with respect to the normalization,
        at given normalization.

        Equivalent to dts(data, bkg, unit_excess, ue_sum, norm, order=1)

        Use Numba to avoid allocating intermediate arrays.

        """

        f = -ue_sum

        for i in prange(len(data)):
            r = unit_excess[i] / (bkg[i] + norm * unit_excess[i])

            dxr = data[i] * r
            f += dxr

        return 2*f

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def d2ts(data, bkg, unit_excess, norm):
        """
        Get the first derivative of TS with respect to the normalization,
        at given normalization.

        Equivalent to dts(data, bkg, unit_excess, ue_sum, norm, order=2)

        Use Numba to avoid allocating intermediate arrays.

        """

        fp = data.dtype.type(0)

        for i in prange(len(data)):
            r = unit_excess[i] / (bkg[i] + norm * unit_excess[i])

            dxr = data[i] * r * r
            fp += dxr

        return -2*fp

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def halley_stepsize(data, bkg, unit_excess, ue_sum, norm):
        """
        Step size for Halley's method.  Equivalent to

        f = self.dts(data, bkg, unit_excess, ue_sum, norm, 1)
        fp = self.dts(data, bkg, unit_excess, ue_sum, norm, 2)
        fpp = self.dts(data, bkg, unit_excess, ue_sum, norm, 3)

        -2 * f * fp / (2 * fp * fp - f * fpp)

        Use Numba to avoid allocating intermediate arrays.

        """

        f = -ue_sum
        fp = data.dtype.type(0)
        fpp = data.dtype.type(0)

        for i in prange(len(data)):
            r = unit_excess[i] / (bkg[i] + norm * unit_excess[i])

            dxr = data[i] * r
            f += dxr
            dxr *= r
            fp += dxr
            dxr *= r
            fpp += dxr

        return (f * fp)/(fp**2 - f * fpp)

    '''
    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def newton_stepsize(data, bkg, unit_excess, ue_sum, norm):
        """
        Step size for Newton's method.  Equivalent to

        f = self.dts(data, bkg, unit_excess, ue_sum, norm, 1)
        fp = self.dts(data, bkg, unit_excess, ue_sum, norm, 2)

        -f/fp

        Use Numba to avoid allocating intermediate arrays.

        """

        f  = -ue_sum
        fp = data.dtype.type(0)

        for i in prange(len(data)):
            r = unit_excess[i] / (bkg[i] + norm * unit_excess[i])

            dxr = data[i] * r
            f += dxr
            dxr *= r
            fp += dxr

        return f/fp
    '''

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def newton_loop(data, bkg, unit_excess, ue_sum, max_iter, conv_frac_tol):

        norm = data.dtype.type(0)
        conv = False

        for iteration in range(max_iter):
            f  = -ue_sum
            fp = data.dtype.type(0)

            for i in prange(len(data)):
                r = unit_excess[i] / (bkg[i] + norm * unit_excess[i])

                dxr = data[i] * r
                f += dxr
                dxr *= r
                fp += dxr

            step = f/fp
            norm += step

            if step < norm * conv_frac_tol:
                conv = True
                break

        return norm, conv, iteration

    def solve(self, data, bkg, unit_excess, ue_sum):
        """
        Compute the Poisson log-likelihood test statistic (TS)
        associated with the best fit of observation data to the
        background model plus a point-source with response
        unit_excess.

        Each of data, bkg, and unit_excess are linear arrays
        representing a subset of the full CDS.  Neither data
        nor bkg may have any zero entries.  The sum of the
        response over the *full* CDS (not just the selected
        subset of cells) must also be provided.

        This function is parallelized over CDS cells (i.e., the
        vector width of data/bkg/unit_excess) using Numba.
        Use numba.set_num_threads() before calling to limit
        the number of processors used.

        Parameters
        ----------
        data : np.ndarray of float
          Observed counts
        bkg : np.ndarray of float
          Background model estimates
        unit_excess : np.ndarray of float
          Expected excess per normalization unit
        ue_sum: float
          Sum of unit excess over *all* cells of CDS (not just
          those referenced in data/bkg/unit_excess

        Returns
        ------
        (float, float, float, bool):
             - LL test statistic,
             - normalization (scale factor for unit_excess, representing
               relative brightness of source)
             - normalization error
             - error status (False = no error, True = error)
             - number of Newton iterations needed to converge

        The normalization error is obtained by approximating the TS
        function as as parabola (i.e., valid in the Gaussian
        regime). TS and norm are indeed valid in the Poisson regime.

        """

        norm = data.dtype.type(0)

        dts0 = self.d1ts(data, bkg, unit_excess, ue_sum, norm)

        if dts0 < 0:
            # Underfluctuation

            # A negative norm is not well defined since we use a
            # Poisson Likelihood. When negative norm is allowed, we
            # assume that we are in the Gaussian regime

            ddts0 = self.d2ts(data, bkg, unit_excess, norm)

            if self.allow_negative:
                ts   = 0.5 * dts0 * dts0 / ddts0
                norm = -dts0 / ddts0
                norm_err = np.sqrt(-2 / ddts0)

            else:
                ts = 0
                norm = 0
                if ddts0 == 0:
                    norm_err = -1/dts0
                else:
                    norm_err = -(np.sqrt(dts0*dts0 - 2*ddts0) + dts0) / ddts0

            # never failed, results were analytical
            return (ts, norm, norm_err, False)

        # Newton's method
        norm, conv, n_iters = self.newton_loop(data, bkg, unit_excess, ue_sum,
                                               self.max_iter, self.conv_frac_tol)

        # One extra step using Halley's method to avoid bias
        norm += self.halley_stepsize(data, bkg, unit_excess, ue_sum, norm)

        # Compute ts, norm error and checks
        ts = self.ts(data, bkg, unit_excess, ue_sum, norm)

        # estimated error in norm
        norm_err = np.sqrt(-2/self.d2ts(data, bkg, unit_excess, norm))

        # sanity checks
        failed = (not conv or norm < 0)
        if failed:
            logger.warning("Failed to converge")

        if ts < 0:
            if ts < -self.zero_ts_tol:
                failed = True
                logger.warning("Failed -- ts is negative")
            else:
                #Assumed to be a numerical error
                ts = 0

        return (ts, norm, norm_err, failed, n_iters)
