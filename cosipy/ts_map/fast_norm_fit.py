
import numpy as np

from numpy import log, sqrt, sqrt
from math import factorial

class FastNormFit:
    """
    Perform a fast poisson maximum likelihood ratio fit of the normalization of a 
    source over background.

    The likelihood ratio as a function of the norm is computed as follow

    .. math::
    
        TS(N) = 2 \\sum_i \\left( \\frac{\\log P(d_i; b_i+N e_i)}{\\log P(d_i; b_i)} \\right)

    where :math:`P(d; \lambda)` is the Poisson probability of obtaining :math:`d`
    count where :math:`\lambda` is expected on average; :math:`b` is the estimated
    number of background counts; :math:`N` is the normalization; and :math:`e` 
    is the expected excess -i.e. signal- per normalization unit -i.e. the 
    number of excess counts equals :math:`N`. 

    It can be shown that :math:`TS(N)` has analytic derivative of arbitrary 
    order and that the Newton's method is guaranteed to converge if initialized
    at :math:`N=0`.

    .. note::

        The background is not allowed to float. It is assumed the error on the
        estimation of the background is small compared to the fluctuation of the
        background itself (i.e. :math:`N_{B}/N_{off} \\lll 1`).

    .. note::
    
        Because of the Poisson probability, :math:`TS(N)` is only well-defined
        for :math:`N \geq 1`. By default, underfluctuations are set to 
        :math:`TS(N=0) = 0`. For cases when there is benefit in letting the
        normalization float to negative values, you can use `allow_negative`, 
        but in that case the results are only valid in the Gaussian regime.

    Args:
        max_iter (int): Maximum number of iteration
        conv_frac_tol (float): Convergence stopping condition, expressed as the
            ratio between the size of the last step and the current norm value.
        zero_ts_tol (float): If zero_ts_tol < TS < 0, then TS is set to 0 
            without failed flag status (analytically, TS < 0 should never happen)
        allow_negative (bool): Allow the normalization to float toward 
            negative values
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
    def ts(data, bkg, unit_excess, norm):
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
            float or array: TS, same shape as norm
        """
        
        data = np.array(data).flatten()
        bkg = np.array(bkg).flatten()
        unit_excess = np.array(unit_excess).flatten()
        
        ts = 0 if np.isscalar(norm) else np.zeros(np.shape(norm))

        for d,b,ue in zip(data, bkg, unit_excess): 

            e = norm*ue

            if d == 0:
                ts -= e
            else:
                ts += d * log((b+e) / b) - e
 
        ts *= 2
     
        return ts

    @staticmethod
    def dts(data, bkg, unit_excess, norm, order=1):
        """
        Get the derivative of TS with respecto to the normalization,
        at given normalization. 

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

        if order < 1:
            raise ValueError("Order must be > 1")
        
        data = np.array(data).flatten()
        bkg = np.array(bkg).flatten()
        unit_excess = np.array(unit_excess).flatten()

        dts = 0 if np.isscalar(norm) else np.zeros(np.shape(norm))

        for d,b,ue in zip(data, bkg, unit_excess): 

            e = norm*ue

            if d > 0:
                dts += d * (ue/(b+e))**order

        if order == 1:

            for ue in unit_excess: 
        
                dts -= ue

        dts *= 2 * ((-1)**(order-1)) * factorial(order-1)

        return dts
        
    def solve(self, data, bkg, unit_excess):
        """
        Get the maximum TS, fitted normalization and normalization error 
        (:math:`\Delta TS = 1`)

        .. note::

            The normalization error is obtained from approximating the TS 
            function as as parabola (i.e. valid in the Gaussian regime). TS
            and norm are indeed valid in the Poisson regime.

        Args:
            data (array): Observed counts
            bkg (array): Background estimation. Same size as data. 
                Every element should be >0
            unit_excess (array): Expected excess per normalization unit. 
                Same size as data.

        Return:
            (float, float, float, bool): ts, norm, norm error and status (0 = good) 
        """
        
        dts0 = self.dts(data, bkg, unit_excess, 0)

        if dts0 < 0:
            # Underfluctuation

            # A negative norm is not well defined since we use a
            # Poisson Likelihood. When negative norm is allowed we assume
            # that we are in the Gaussian regime

            ddts0 = self.dts(data, bkg, unit_excess, 0, order = 2)

            if self.allow_negative:

                ts = dts0*dts0/2/ddts0

                norm = -dts0/ddts0

                norm_err = sqrt(-2/ddts0)
            
            else:

                ts = 0
                norm = 0

                if ddts0 == 0:

                    norm_err = -1/dts0
                
                else:

                    norm_err = -(sqrt(dts0*dts0 - 2*ddts0) + dts0) / ddts0

            #Never failed, results where analytical
            failed = False
            
            return (ts, norm, norm_err, failed)

        # Newton's method

        conv = False
        norm = 0
        iteration = 0
        
        while (not conv and iteration < self.max_iter):

            step = (-self.dts(data, bkg, unit_excess, norm, order = 1) /
                    self.dts(data, bkg, unit_excess, norm, order = 2)
)
        
            norm += step

            if step/norm < self.conv_frac_tol:
                conv = True

            iteration += 1

        # One extra step using Halley's method to avoid being biased toward 
            
        f = self.dts(data, bkg, unit_excess, norm,1)
        fp = self.dts(data, bkg, unit_excess, norm,2)
        fpp = self.dts(data, bkg, unit_excess, norm,3)
        norm -= 2 * f * fp / (2 * fp * fp - f * fpp)

        # Compute ts, norm error and checks

        ts = self.ts(data, bkg, unit_excess, norm)

        norm_err = sqrt(-2/self.dts(data, bkg, unit_excess, norm, 2))

        failed = (norm < 0  or iteration == self.max_iter)

        if ts < -self.zero_ts_tol:
            failed = True
        elif ts < 0:
            #Assumed to be a numerical error
            ts = 0

        return (ts, norm, norm_err, failed, iteration)
