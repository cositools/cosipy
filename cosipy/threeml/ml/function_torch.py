import torch
import numpy as np
from astromodels.functions.function import (
    Function1D,
    Function2D,
    Function3D,
    FunctionMeta,
    ModelAssertionViolation,
)

import astropy.units as astropy_units


import logging
logger = logging.getLogger(__name__)

class FastPowerlawPyTorch(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        A simple power-law

    latex : $ K~\left(\frac{x}{piv}\right)^{index} $

    parameters :

        K :

            desc : Normalization (differential flux at the pivot value)
            initial value : 1.0
            is_normalization : True
            transformation : log10
            min : 1e-30
            max : 1e3
            delta : 0.1

        piv :

            desc : Pivot value
            initial value : 1
            fix : yes

        index :

            desc : Photon index
            initial value : -2.01
            min : -10
            max : 10
      
    properties:
        devices:
            desc: devices used by torch
            initial value: cpu
    """
    
    def _set_units(self, x_unit, y_unit):
        self.index.unit = astropy_units.dimensionless_unscaled

        # The pivot energy has always the same dimension as the x variable
        self.piv.unit = x_unit

        # The normalization has the same units as the y

        self.K.unit = y_unit
        
    
    def evaluate(self, x, K, piv, index):
        

        if isinstance(x, astropy_units.Quantity):
            x_ = x.value
        else:
            x_ = x

        if isinstance(K, astropy_units.Quantity):
            K_, piv_, index_ = K.value, piv.value, index.value
            unit_ = self.y_unit
        else:
            K_, piv_, index_ = K, piv, index
            unit_ = 1.0

        x_t = torch.as_tensor(x_, dtype=torch.float64, device=self.devices.value)
        x_t = x_t.view(-1, 1)
        K_, piv_, index_ = [torch.as_tensor(t, dtype=torch.float64, device=self.devices.value) for t in (K_, piv_, index_)]
        res = torch.div(x_t, piv_)
        res.pow_(index_)
        res.mul_(K_)
        
        if unit_ == 1.0:
            return res.cpu().numpy()
        else:
            return res.cpu().numpy() * unit_



class FastGaussianPyTorch(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        A Gaussian function

    latex : $ F \frac{1}{\sigma \sqrt{2 \pi}}\exp{\frac{(x-\mu)^2}{2~(\sigma)^2}} $

    parameters :

        F :

            desc : Integral between -inf and +inf. Fix this to 1 to obtain a Normal
                    distribution
            initial value : 1

        mu :

            desc : Central value
            initial value : 0.0

        sigma :

            desc : standard deviation
            initial value : 1.0
            min : 1e-12
    
    properties:
        devices:
            desc: devices used by torch
            initial value: cpu
    tests :
        - { x : 0.0, function value: 0.3989422804014327, tolerance: 1e-10}
        - { x : -1.0, function value: 0.24197072451914337, tolerance: 1e-9}

    """
    #Place this here to avoid recomputing it all the time
    __norm_const = 1.0 / np.sqrt(2 * np.pi)

    def _set_units(self, x_unit, y_unit):

        # The normalization is the integral from -inf to +inf, i.e., has dimensions of
        # y_unit * x_unit
        self.F.unit = y_unit * x_unit

        # The mu has the same dimensions as the x
        self.mu.unit = x_unit

        # sigma has the same dimensions as x
        self.sigma.unit = x_unit

    
    def evaluate(self, x, F, mu, sigma):
        
        
        norm = self.__norm_const / sigma
        x = torch.as_tensor(x, dtype=torch.float64, device=self.devices.value)
        mu = torch.as_tensor(mu, dtype=torch.float64, device=self.devices.value)
        sigma = torch.as_tensor(sigma, dtype=torch.float64, device=self.devices.value)

        result = F * norm * torch.exp(-torch.pow(x - mu, 2.0) / (2 * torch.pow(sigma, 2.0)))
        
        return result.cpu().numpy()




class FastCutoffPowerlawPyTorch(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        A power law multiplied by an exponential cutoff

    latex : $ K~\left(\frac{x}{piv}\right)^{index}~\exp{-x/xc} $

    parameters :

        K :

            desc : Normalization (differential flux at the pivot value)
            initial value : 1.0
            is_normalization : True
            transformation : log10
            min : 1e-30
            max : 1e3
            delta : 0.1

        piv :

            desc : Pivot value
            initial value : 1
            fix : yes

        index :

            desc : Photon index
            initial value : -2
            min : -10
            max : 10

        xc :

            desc : Cutoff energy
            initial value : 10.0
            transformation : log10
            min: 1.0
    
    properties:
        devices:
            desc: devices used by torch
            initial value: cpu
    """

    def _set_units(self, x_unit, y_unit):
        # The index is always dimensionless
        self.index.unit = astropy_units.dimensionless_unscaled

        # The pivot energy has always the same dimension as the x variable
        self.piv.unit = x_unit

        self.xc.unit = x_unit

        # The normalization has the same units as the y

        self.K.unit = y_unit
        
        
    # noinspectionq PyPep8Naming
    def evaluate(self, x, K, piv, index, xc):

        if isinstance(x, astropy_units.Quantity):
            index_ = index.value
            K_ = K.value
            piv_ = piv.value
            xc_ = xc.value
            x_ = x.value

            unit_ = self.y_unit

        else:
            unit_ = 1.0
            K_, piv_, x_, index_, xc_ = K, piv, x, index, xc

        x_t = torch.as_tensor(x_, dtype=torch.float64, device=self.devices.value)
        x_t = x_t.view(-1, 1)
        K_, piv_, index_, xc_ = [torch.as_tensor(t, dtype=torch.float64, device=self.devices.value) for t in (K_, piv_, index_, xc_)]
        res = torch.div(x_t, piv_)
        res.pow_(index_)
        res.mul_(K_)
        res.mul_(torch.exp_(-x_t/xc_))        

        if unit_ == 1.0:
            return res.cpu().numpy()
        else:
            return res.cpu().numpy() * unit_
            
class FastSuperCutoffPowerlawPyTorch(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        A power law with a super-exponential cutoff

    latex : $ K~\left(\frac{x}{piv}\right)^{index}~\exp{(-x/xc)^{\gamma}} $

    parameters :

        K :

            desc : Normalization (differential flux at the pivot value)
            initial value : 1.0
            min : 1e-50
            is_normalization : True
            transformation : log10

        piv :

            desc : Pivot value
            initial value : 1
            fix : yes

        index :

            desc : Photon index
            initial value : -2
            min : -10
            max : 10

        xc :

            desc : Cutoff energy
            initial value : 10.0
            min : 1.0
            transformation : log10

        gamma :

            desc : Index of the super-exponential cutoff
            initial value : 1.0
            min : 0.1
            max : 10.0
    
    properties:
        devices:
            desc: devices used by torch
            initial value: cpu
    """

    def _set_units(self, x_unit, y_unit):
        # The index is always dimensionless
        self.index.unit = astropy_units.dimensionless_unscaled
        self.gamma.unit = astropy_units.dimensionless_unscaled

        # The pivot energy has always the same dimension as the x variable
        self.piv.unit = x_unit

        # The cutoff has the same dimensions as x
        self.xc.unit = x_unit

        # The normalization has the same units as the y

        self.K.unit = y_unit

    # noinspection PyPep8Naming
    def evaluate(self, x, K, piv, index, xc, gamma):

        if isinstance(x, astropy_units.Quantity):
            index_ = index.value
            K_ = K.value
            piv_ = piv.value
            xc_ = xc.value
            gamma_ = gamma.value
            x_ = x.value

            unit_ = self.y_unit

        else:
            unit_ = 1.0
            K_, piv_, x_, index_, xc_, gamma_ = K, piv, x, index, xc, gamma

        x_t = torch.as_tensor(x_, dtype=torch.float64, device=self.devices.value)
        x_t = x_t.view(-1, 1)
        K_, piv_, index_, xc_, gamma_ = [torch.as_tensor(t, dtype=torch.float64, device=self.devices.value) for t in (K_, piv_, index_, xc_, gamma_)]
        
        res = torch.exp( index_ * torch.log(x_t / piv_) - torch.pow(x_t / xc_, gamma_) )
        res.mul_(K_)
        
        return res.cpu().numpy() * unit_
