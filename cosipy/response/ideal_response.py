import warnings
from collections.abc import Callable
from typing import Iterable, Tuple, Union, Iterator

import numpy as np
from scipy.stats import rv_continuous, truncnorm, norm, uniform, poisson
from scipy.stats.sampling import SimpleRatioUniforms
from scipy.special import erfi, erf

from astropy.units import Quantity
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord

from scoords import SpacecraftFrame

from cosipy.util.iterables import itertools_batched

from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventInSCFrame
from cosipy.polarization import (
    StereographicConvention,
    PolarizationConvention,
    PolarizationAngle
)

from cosipy.response.relative_coordinates import RelativeCDSCoordinates

from cosipy.interfaces import ExpectationDensityInterface
from cosipy.interfaces.data_interface import EmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import EmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import (
    FarFieldInstrumentResponseFunctionInterface,
    FarFieldSpectralInstrumentResponseFunctionInterface,
    FarFieldSpectralPolarizedInstrumentResponseFunctionInterface
)

from cosipy.interfaces.photon_parameters import (
    PhotonInterface,
    PhotonWithDirectionAndEnergyInSCFrameInterface,
    PhotonWithEnergyInterface,
    PhotonWithDirectionInSCFrameInterface,
    PhotonListWithDirectionAndEnergyInSCFrameInterface
)
from cosipy.response.photon_types import (
    PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface as PolDirESCPhoton,
    PolarizedPhotonListWithDirectionAndEnergyInSCFrameStereographicConventionInterface as PolDirESCPhotonList,
    PhotonWithDirectionAndEnergyInSCFrame,
    PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention
)


def _to_rad(angle):
    if isinstance(angle, (Quantity, Angle)):
        return angle.to_value(u.rad)
    else:
        return angle


class _SimpleRVSMixin:
    """
    Helper mixin for custom distributions (rv_continuous subclasses)
    using SimpleRatioUniforms

    Subclasses need to define _pdf

    """

    @property
    def _mode(self):
        # Return analytic mode if you can.
        # Otherwise it will be estimated numerically
        return None

    def _simple_ratio_uniforms_rvs(self, *args, size=None, random_state=None):
        if warnings.catch_warnings():
            # Suppress warning
            # "WARNING RuntimeWarning: [objid: SROU] 22 : mode: try
            # finding it (numerically) => (distribution) incomplete
            # distribution object, entry missing"
            # when the mode need to be computed analytically

            if self._mode is None:
                warnings.filterwarnings(
                    "ignore",
                    message=r".*\[objid: SROU\].*",
                    category=RuntimeWarning,
                )

            rng = SimpleRatioUniforms(self, random_state=random_state, mode=self._mode)

        if isinstance(size, tuple) and not size: # == ()
            # SimpleRatioUniforms.rvs expects an integer, tuple of
            # integers or None.  It crashes with an empty tuple, which
            # corresponds to a scalar.
            size = None

        return rng.rvs(size=size)

    def _rvs(self, *args, **kwargs):
        return self._simple_ratio_uniforms_rvs(*args, **kwargs)

class _RVSMixin(_SimpleRVSMixin):
    """
    Helper mixin for custom distributions (rv_continuous subclasses)
    that will likely only get a sample per setup

    Subclasses need to define _pdf and _cdf
    """

    def _rvs(self, *args, size=None, **kwargs):

        # Faster than default _rvs for large sizes, but slow setup
        # Most of the time we'll need a new setup per energy

        if size is None or size == tuple():
            return super()._rvs(*args, size=size, **kwargs)
        else:
            return self._simple_ratio_uniforms_rvs(*args, size = size, **kwargs)

class KleinNishinaPolarScatteringAngleDist(_RVSMixin, rv_continuous):
    """
    Klein-Nishina scattering angle distribution
    """

    def __init__(self, energy, *args, **kwargs):

        super().__init__(0, *args, a=0, b=np.pi, **kwargs)

        self._eps = energy.to_value(u.keV) / 510.99895069  # E/m_ec^2

        # Normalization
        # Mathematica
        # Integrate[(
        #  Sin[\[Theta]] (1 + 1/(
        #     1 + \[Epsilon] (1 - Cos[\[Theta]])) + \[Epsilon] (1 -
        #        Cos[\[Theta]]) -
        #     Sin[\[Theta]]^2))/(1 + \[Epsilon] (1 -
        #       Cos[\[Theta]]))^2, {\[Theta], 0, \[Pi]},
        #  Assumptions -> {\[Epsilon] > 0}]

        A = 2 * self._eps * (2 + self._eps * (1 + self._eps) * (8 + self._eps)) / (1 + 2 * self._eps) ** 2
        B = (-2 + self._eps * (self._eps - 2)) * np.log(1 + 2 * self._eps)

        self._norm = (A + B) / self._eps ** 3

    def _pdf(self, phi, *args):

        # Substitute Compton kinematic equation in Klein-Nishina dsigma/dOmega
        # Mathematica
        # eratio = 1 + self._eps (1 - cos_phi]) (*e/ep*)
        # (1/eratio)^2 (1/eratio + eratio - Sin[\[Theta]]^2) Sin[\[Theta]]

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        A = 1 + (1 / (1 + self._eps * (1 - cos_phi))) + self._eps * (1 - cos_phi) - sin_phi ** 2
        B = (1 + self._eps * (1 - cos_phi)) ** 2

        # Extra sin(phi) to account for phasespace
        return sin_phi * A / B / self._norm

    def _cdf(self, phi, *args):

        # Mathematica
        # Integrate[(
        #  Sin[\[Theta]] (1 + 1/(
        #     1 + self._eps (1 - cos_phi])) + self._eps (1 -
        #        cos_phi]) -
        #     Sin[\[Theta]]^2))/ ((1 + self._eps (1 -
        #       cos_phi]))^2), {\[Theta], 0, \[Theta]p},
        #  Assumptions -> {self._eps > 0, \[Theta]p < \[Pi], \[Theta]p  > 0}]

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        eps = self._eps
        eps2 = eps * eps
        eps3 = eps2 * eps

        A = 1 + eps - eps * cos_phi
        logA = np.log(A)
        B = (eps * (4 + 10 * eps + 8 * eps2 + eps3) \
             - 2 * eps3 * cos_phi ** 3 \
             + 2 * (1 + eps) ** 2 * (-2 - 2 * eps + eps2) * logA \
             + eps2 * cos_phi ** 2 * (6 + 10 * eps + eps2 + 2 * (-2 - 2 * eps + eps2) * logA) \
             - 2 * eps * cos_phi * (2 + 8 * eps + 8 * eps2 + eps3 \
                                    + 2 * (-2 - 4 * eps - eps2 + eps3) * logA))
        C = 2 * eps3 * A * A

        return B / C / self._norm

class KleinNishinaAzimuthalScatteringAngleDist(_RVSMixin, rv_continuous):

    def __init__(self, energy, theta, *args, **kwargs):
        """
        Conditional probability, given a polar angle and energy.

        NOTE: input phi in pdf(phi) and cdf(phi) MUST lie between
        [0,2*pi]. The results are unpredictable otherwise.

        Parameters
        ----------
        energy
        theta: polar angle
        args
        kwargs

        """

        super().__init__(0, *args, a=0, b=2*np.pi, **kwargs)

        theta = _to_rad(theta)

        # precompute some stuff
        self._eps = energy.to_value(u.keV) / 510.99895069  # E/m_ec^2
        self._sin_theta2 = np.sin(theta) ** 2
        self._energy_ratio =  1 + self._eps * (1 - np.cos(theta)) # From kinematics
        self._energy_ratio2 = self._energy_ratio * self._energy_ratio
        self._energy_ratio_inv = 1/self._energy_ratio
        self._energy_ratio_inv2 = self._energy_ratio_inv * self._energy_ratio_inv
        self._energy_ratio_inv3 = self._energy_ratio_inv2 * self._energy_ratio_inv

        # Mathematica
        # Integrate[(1/eratio + eratio - 2 sintheta2 Cos[\[Phi]]^2)/
        #   eratio^2, {\[Phi], 0, 2 \[Pi]},
        #   Assumptions -> {\[Epsilon] > 0}] // FullSimplify
        self._norm = 2 * np.pi * (1 + self._energy_ratio2 - self._sin_theta2 * self._energy_ratio) * self._energy_ratio_inv3

    def _pdf(self, phi, *args):
        """
        Parameters
        ----------
        phi: azimuthal angle, starting from the electric field vector
        direction args

        """

        phi = _to_rad(phi)

        cos_phi = np.cos(phi)

        return (self._energy_ratio + self._energy_ratio_inv - 2 * self._sin_theta2 * cos_phi * cos_phi) * self._energy_ratio_inv2 / self._norm

    def _cdf(self, phi, *args):

        phi = _to_rad(phi)

        # Mathematica
        # Integrate[(1/eratio + eratio - 2 sintheta2 Cos[\[Phi]]^2)/
        #   eratio^2, {\[Phi], 0, \[Phi]lim},
        #   Assumptions -> {\[Epsilon] > 0}] // FullSimplify

        A = phi + phi*self._energy_ratio2 - self._energy_ratio * self._sin_theta2 * phi - self._energy_ratio * self._sin_theta2 * np.cos(phi) * np.sin(phi)

        return A * self._energy_ratio_inv3 / self._norm

class ARMNormDist(_SimpleRVSMixin, rv_continuous):

    def __init__(self, phi, angres, *args, **kwargs):
        """
        This accounts for the truncating effect since ARM is  limited to [-phi, pi-phi].
        It also accounts for the sin(phi+arm) phasespace

        Parameters
        ----------
        phi: Polar scattering angle
        angres: Standard deviation of the equivalent gaussian
        args
        kwargs
        """

        phi = _to_rad(phi)
        angres = _to_rad(angres)

        super().__init__(0, *args, a=-phi, b= np.pi - phi, **kwargs)

        # normalized such that int_0^pi random_arm = 1  (already includes sin(phi+arm))
        # Integrate[PDF[TruncatedDistribution[{0,\[Pi]},NormalDistribution [\[Phi],\[Sigma]]], x]Sin[x],{x,0,\[Pi]}]//Re//FullSimplify
        # Mathematica couldn't get only the real part analytically

        self._phi = phi
        self._angres = angres

        self._norm = np.real(
            np.exp(-(angres ** 2 / 2) - 1j * phi) *
            (1j * erf((np.pi + 1j * angres ** 2 - phi) / (np.sqrt(2) * angres)) +
             np.exp(2j * phi) * (erfi((angres ** 2 - 1j * phi) / (np.sqrt(2) * angres)) -
                                 erfi((1j * np.pi + angres ** 2 - 1j * phi) / (np.sqrt(2) * angres))) +
             erfi((angres ** 2 + 1j * phi) / (np.sqrt(2) * angres)))
            / (2 * (erf(phi / (np.sqrt(2) * angres)) - erf((-np.pi + phi) / (np.sqrt(2) * angres)))))

        self._truncnorm_dist = truncnorm(-self._phi / self._angres, (np.pi - self._phi) / self._angres, 0, self._angres)

    def _pdf(self, arm, *args):

        return self._truncnorm_dist.pdf(arm) * np.sin(self._phi + arm) / self._norm

class ARMMultiNormDist(rv_continuous):

    def __init__(self, phi, angres, angres_weights, *args, **kwargs):
        """
        Describe the ARM distribution by a combination of multiple
        [truncated] gaussians

        Parameters
        ----------
        phi
        angres
        angres_weights
        args
        kwargs

        """

        phi = _to_rad(phi)
        angres = _to_rad(angres)

        super().__init__(0, *args, a=-phi, b= np.pi - phi, **kwargs)

        angres = np.atleast_1d(angres)

        weights = np.broadcast_to(angres_weights, angres.shape)
        self._weights = weights / np.sum(weights)
        self._dists = [ARMNormDist(phi, res) for res in angres]

    def _pdf(self, arm, *args):

        prob = np.zeros(np.shape(arm))

        for w,dist in zip(self._weights,self._dists):
            prob += w*dist._pdf(arm)

        return prob

    def _rvs(self, *args, size=None, random_state=None):

        if random_state is None:
            random_state = self.random_state

        samples = np.empty(size)

        idx = random_state.choice(np.arange(len(self._dists)), size = size, p = self._weights)

        for i in range(len(self._dists)):

            dist = self._dists[i]

            mask = idx == i

            nmask = np.count_nonzero(mask)

            samples[mask] = dist._rvs(size = nmask)

        return samples

class ThresholdKleinNishinaPolarScatteringAngleDist(KleinNishinaPolarScatteringAngleDist):

    def __init__(self, energy, energy_threshold=None, *args, **kwargs):

        super().__init__(energy)

        if energy_threshold is None:
            self._renormalizable = True
            self._renormalizable_error = None
            self._min_phi = 0
        else:

            # Mathematica
            # Solve[e/(e - edepmax) == 1 + \[Epsilon] (1 - (-1)), edepmax]

            max_energy_deposited = 2 * energy * self._eps / (1 + 2 * self._eps)

            if energy_threshold > max_energy_deposited:
                self._renormalizable = False
                self._renormalizable_error = ValueError(
                f"Threshold ({energy_threshold}) is greater than the maximum possible deposited energy ({max_energy_deposited}). PDF cannot be normalized")
            else:
                self._renormalizable = True
                self._renormalizable_error = None

                # Mathematica
                # Solve[e/(e - ethresh) ==
                #   1 + \[Epsilon] (1 - Cos[\[Theta]]), \[Theta] ]

                energy_threshold = energy_threshold.to_value(energy.unit)
                energy = energy.value

                eps_ediff = self._eps * (energy - energy_threshold)

                self._min_phi = np.arccos((eps_ediff - energy_threshold) / eps_ediff)

        # Renormalize
        self._cdf_min_phi = None
        self._norm_factor = None

        if self._renormalizable:
            self._cdf_min_phi = super()._cdf(self._min_phi)
            self._norm_factor = 1 / (1 - self._cdf_min_phi)

    def _renormalize(self, phi, prob):

        if np.isscalar(phi):
            if phi < self._min_phi:
                prob = 0
        else:
            prob = np.asarray(prob)
            phi = np.asarray(phi)
            prob[phi < self._min_phi] = 0

        prob *= self._norm_factor

        return prob

    def _pdf(self, phi, *args):

        if not self._renormalizable:
            # While the PDF can't be normalized,
            # and there we can't have a CDF or RVS,
            # we can still return the probability = 0
            # to prevent other code from crashing
            return np.zeros_like(phi)

        phi = _to_rad(phi)

        prob = super()._pdf(phi, *args)

        return self._renormalize(phi, prob)

    def _cdf(self, phi, *args):

        if not self._renormalizable:
            raise self._renormalizable_error

        phi = _to_rad(phi)

        cum_prob = super()._cdf(phi, *args) - self._cdf_min_phi

        return self._renormalize(phi, cum_prob)

    def _rvs(self, *args, **kwargs):
        if not self._renormalizable:
            raise self._renormalizable_error

        return super()._rvs(*args, **kwargs)

class MeasuredEnergyDist(rv_continuous):

    def __init__(self, energy, energy_res, phi, full_absorp_prob, *args, **kwargs):
        """
        This is a *conditional* probability. We will assume the
        uncertainty on the measured angle phi is 0 (all the CDS errors
        will come from the ARM distribution)

        If it is fully absorbed, then the deposited energy equal the
        initial energy.

        If it escaped, then it will assume that the deposited energy
        corresponds to the energy of the first hit, following the
        Compton equation

        The measured energy will be drawn from a normal distribution
        centered at the deposited energy and std equal to
        energy_deposited*energy_res

        The geometry was not taking into account for the backscatter
        criterion since it was too complicated.

        Inputs and outputs are values assumed to be in the same units
        as input energy.

        Parameters
        ----------
        energy: initial energy.
        energy_res: function returning the energy resolution function of
           energy. Both input and output have energy units
        phi: polar scattered angle
        full_absorp_prob: probability of landing in the photopeak
        args
        kwargs

        """

        super().__init__(0, *args, a=0, **kwargs)

        if full_absorp_prob < 0 or full_absorp_prob > 1:
            raise ValueError(f"full_absorp_prob must be between [0,1]. Got {full_absorp_prob}")

        eps = (energy / Quantity(510.99895069, u.keV)).value

        phi = _to_rad(phi)
        energy_deposited = energy * (1 - 1 / (1 + eps * (1 - np.cos(phi))))

        self._full_prob = full_absorp_prob
        self._partial_prob = 1 - full_absorp_prob

        self._dist_full = norm(loc=energy.value, scale = energy_res(energy).to_value(energy.unit))
        self._dist_partial = norm(loc=energy_deposited.value, scale =  energy_res(energy_deposited).to_value(energy.unit))

    def _pdf(self, measured_energy, *args):
        return self._full_prob * self._dist_full.pdf(measured_energy) + self._partial_prob * self._dist_partial.pdf(measured_energy)

    def _cdf(self, measured_energy, *args):
        return self._full_prob * self._dist_full.cdf(measured_energy) + self._partial_prob * self._dist_partial.cdf(measured_energy)

    def _rvs(self, *args, size=None, random_state=None):

        full_absorp = uniform.rvs(size=size, random_state = random_state)  < self._full_prob

        nfull = np.count_nonzero(full_absorp)
        npartial = full_absorp.size - nfull

        samples = np.empty(full_absorp.shape)

        samples[full_absorp] = self._dist_full.rvs(*args, size=nfull, random_state=random_state)
        samples[np.logical_not(full_absorp)] = self._dist_partial.rvs(*args, size=npartial, random_state=random_state)

        return samples

class LogGaussianCosThetaEffectiveArea:

    def __init__(self,
                 max_area:Quantity,
                 max_area_energy:Quantity,
                 sigma_decades: float,
                 batch_size = 1000):
        """
        The effective area is represented as a log-gaussian as function of
        energy and a cos(theta) dependence as a function of the
        instrument colatitude theta.  =0 beyond theta = 90 deg

        Parameters
        ----------
        max_area: maximum effective area
        max_area_energy: energy where the effective area peaks
        sigma_decades:

        """

        self._max_area = max_area
        self._max_area_energy = max_area_energy.to_value(u.keV)
        self._sigma_decades = sigma_decades

        self._batch_size = batch_size

    def __call__(self, photons = Iterable[PhotonWithDirectionAndEnergyInSCFrameInterface]) -> Iterable[Quantity]:
        """
        """

        for batch in itertools_batched(photons, self._batch_size):

            energy = []
            latitude = []

            for photon in batch:

                energy.append(photon.energy_keV)
                latitude.append(photon.direction_lat_rad_sc)

            energy = np.asarray(energy)
            latitude = np.asarray(latitude)

            area = self._max_area * np.exp(-np.log10(energy / self._max_area_energy) ** 2 / 2 / self._sigma_decades / self._sigma_decades)

            area *= np.sin(latitude)
            area[latitude < 0] = 0

            yield from area

class ConstantFractEnergyRes:

    def __init__(self, energy_res):
        """

        Parameters
        ----------
        energy_res: fraction
        """

        self._energy_res = energy_res

    def __call__(self, energy) -> Quantity:
        """
        """

        return self._energy_res * energy

class ConstantAngularResolution:

    def __init__(self, angres, weights = None):
        self._angres = np.atleast_1d(angres)

        if weights is None:
            weights = np.ones(self._angres.size)

        self._weights = weights / np.sum(weights)

    def __call__(self, photons=Iterable[PhotonWithDirectionInSCFrameInterface]) -> Iterable[Quantity]:
        for _ in photons:
            yield self._angres, self._weights

class ConstantTimesExponentialCutoffFullAbsorption:

    def __init__(self, base:float, cutoff_energy:Quantity, batch_size = 1000):
        self._base = base
        self._cutoff_energy = cutoff_energy.to_value(u.keV)
        self._batch_size = batch_size

    def __call__(self, photons = Iterable[PhotonWithEnergyInterface]) -> Iterable[Quantity]:
        """
        """

        for batch in itertools_batched(photons, self._batch_size):

            energy = np.asarray([photon.energy_keV for photon in batch])

            prob = self._base * np.exp(-energy / self._cutoff_energy)

            yield from prob

class UnpolarizedIdealComptonIRF(FarFieldSpectralInstrumentResponseFunctionInterface):

    # The photon class and event class that the IRF implementation can handle
    photon_list_type = PhotonListWithDirectionAndEnergyInSCFrameInterface
    event_data_type = EmCDSEventDataInSCFrameInterface

    def __init__(self,
                 effective_area:Callable[[Iterable[PhotonInterface]], Quantity],
                 energy_resolution:Callable[[Quantity], Quantity],
                 angular_resolution:Callable[[PhotonInterface], Tuple[Quantity, np.ndarray[float]]],
                 full_absorption_prob:Callable[[Iterable[PhotonInterface]], Quantity],
                 energy_threshold:Union[None, Quantity] = None
                 ):

        self._effective_area = effective_area
        self._energy_resolution = energy_resolution
        self._angular_resolution = angular_resolution
        self._full_prob = full_absorption_prob

        if energy_threshold is None:
            self.energy_threshold = 0*u.keV
        else:
            self._energy_threshold = energy_threshold

        self._pol_convention = StereographicConvention()

    @classmethod
    def cosi_like(cls,
                  max_area = 110 * u.cm * u.cm,
                  max_area_energy = 1500 * u.keV,
                  sigma_decades = 0.4,
                  energy_resolution = 0.01,
                  angres = 3*u.deg,
                  angres_fact = [1 / 3., 1, 3, 9],
                  angres_weights = [1, 4, 10, 20],
                  full_absorption_constant = 0.5,
                  full_absorption_exp_cutoff = 10*u.MeV,
                  energy_threshold = 20*u.keV):
        """
        Similar performance as COSI. Meant for code development, not
        science or sensitivity predictions.

        """

        # This angres_fact give a FWHM approx = angres, but with long tails
        max_area = 110 * u.cm * u.cm                if max_area is None else max_area
        max_area_energy = 1500 * u.keV              if max_area_energy is None else max_area_energy
        sigma_decades = 0.4                         if sigma_decades is None else sigma_decades
        energy_resolution = 0.01                    if energy_resolution is None else energy_resolution
        angres = 3 * u.deg                          if angres is None else angres
        angres_fact = np.asarray([1/3.,1,3,9,27])/3 if angres_fact is None else angres_fact
        angres_weights = np.asarray([1,4,5,20,30])  if angres_weights is None else angres_weights
        full_absorption_constant = 0.7              if full_absorption_constant is None else full_absorption_constant
        full_absorption_exp_cutoff = 10 * u.MeV     if full_absorption_exp_cutoff is None else full_absorption_exp_cutoff
        energy_threshold = 20 * u.keV               if energy_threshold is None else energy_threshold

        angres_fact = np.asarray(angres_fact)
        angres_weights = np.asarray(angres_weights)

        effective_area = LogGaussianCosThetaEffectiveArea(max_area, max_area_energy, sigma_decades)
        energy_resolution = ConstantFractEnergyRes(energy_resolution)
        angular_resolution = ConstantAngularResolution(angres * angres_fact, angres_weights)
        full_absorption_prob = ConstantTimesExponentialCutoffFullAbsorption(full_absorption_constant, full_absorption_exp_cutoff)

        return cls(effective_area,
                   energy_resolution,
                   angular_resolution,
                   full_absorption_prob,
                   energy_threshold)

    def _az_prob(self, photon, phi, az):
        return 1/2/np.pi

    def _random_az(self, photon, phi):
        return 2*np.pi*uniform.rvs()

    def _event_probability_const_phi(self, photon:PolDirESCPhoton,
                                     phi:float,
                                     events:Iterable[EmCDSEventInSCFrameInterface]) -> Iterable[float]:
        """
        Computes the probability for a given set of photon parameters, and
        for all events with the same phi

        Note: it is assumed that all events have the same phi!!!

        """

        # Get some needed values from this query
        photon_energy_keV = photon.energy_keV
        photon_energy = Quantity(photon_energy_keV, u.keV, copy = None)
        measured_energy_keV = np.asarray([event.energy_keV for event in events])
        full_absorp_prob = next(self._full_prob([photon]))
        angres, weights = next(self._angular_resolution([photon]))
        psichi_lon = [event.scattered_lon_rad_sc for event in events]
        psichi_lat = [event.scattered_lat_rad_sc for event in events]
        psichi = SkyCoord(lon = psichi_lon, lat = psichi_lat, unit = u.rad, frame = SpacecraftFrame())

        # Convert CDF to relative
        phi_geom, az = RelativeCDSCoordinates(photon.direction, self._pol_convention).to_relative(psichi)

        # Get probability
        # We're assuming the phi measured from kinematics has no
        # errors. Otherwise, the calculation became too complex
        # All directional error come from the uncertainty on psichi
        # (through the ARM, in psichi_geom)
        # P(phi|Ei) * P(Em | Ei, phi) * P(psichi | phi, Ei, PA)
        # P(psichi | phi, Ei, PA) = P(arm | phi) * P(az | phi, Ei)

        prob = ThresholdKleinNishinaPolarScatteringAngleDist(photon_energy, self._energy_threshold).pdf(phi)
        prob *= MeasuredEnergyDist(photon_energy, self._energy_resolution, phi, full_absorp_prob).pdf(measured_energy_keV)
        prob *= ARMMultiNormDist(phi, angres, weights).pdf(phi_geom.rad - phi)
        prob *= self._az_prob(photon, phi, az.rad)

        return prob

    def _event_probability(self, photons: PolDirESCPhotonList, events:EmCDSEventDataInSCFrameInterface) -> Iterable[float]:
        """
        Return the probability density of measuring a given event given a
        photon.

        The units of the output the inverse of the phase space of the
        class event_type data space.  e.g. if the event measured
        photon_energy in keV, the units of output of this function are
        implicitly 1/keV

        NOTE: this implementation runs fast if you sort the queries by
        photon, following by the event phi.

        """

        # This allows to sample the PDF for multiple values at once
        # Multiple event with the phi pretty much only happen during
        # testing though, since for real data the same measured values
        # will not be repeating
        last_photon = None
        last_phi = None
        cached_events = []

        for photon,event in zip(photons, events):

            phi = event.scattering_angle_rad

            if last_photon is None:
                # This only happens for the first event
                last_photon = photon
                last_phi = phi
                cached_events = [event]
                continue

            if photon is last_photon:
                # We can keep caching values, unless phi changed

                if last_phi is phi:
                    # Same photon and phi. Keep caching events
                    cached_events.append(event)
                else:
                    # It's no longer the same. We now need to evaluate
                    # and yield what we have so far
                    yield from self._event_probability_const_phi(last_photon, last_phi, cached_events)

                    # Restart
                    last_photon = photon
                    last_phi = phi
                    cached_events = [event]

            else:
                # It's no longer the same. We now need to evaluate and
                # yield what we have so far
                yield from self._event_probability_const_phi(last_photon, last_phi, cached_events)

                # Restart
                last_photon = photon
                last_phi = phi
                cached_events = [event]

        # Yield the probability for the leftover events
        yield from self._event_probability_const_phi(last_photon, last_phi, cached_events)

    def _random_events(self, photons: PolDirESCPhotonList) -> EmCDSEventDataInSCFrameInterface:
        return self.event_data_type.fromiter(self._random_events_iter(photons), photons.nphotons)

    def _random_events_iter(self, photons: PolDirESCPhotonList) -> Iterable[EmCDSEventInSCFrameInterface]:
        """
        Return a stream of random events, photon by photon.
        """

        for photon in photons:

            energy = photon.energy
            full_absorp_prob = next(self._full_prob([photon]))

            # Random polar (phi) and azimuthal angle from Klein Nishina
            phi = ThresholdKleinNishinaPolarScatteringAngleDist(energy, self._energy_threshold).rvs()
            azimuth = self._random_az(photon, phi)

            # Get the measured energy based on phi and the energy
            # resolution and absroption probabity for the photon
            # location
            measured_energy = MeasuredEnergyDist(energy, self._energy_resolution, phi, full_absorp_prob).rvs()
            measured_energy_keV = Quantity(measured_energy, energy.unit, copy=None).to_value(u.keV)

            # Get a random ARM
            angres, weights = next(self._angular_resolution([photon]))
            arm = ARMMultiNormDist(phi, angres, weights).rvs()

            # Transform arm and az to psichi
            psichi = RelativeCDSCoordinates(photon.direction, self._pol_convention).to_cds(phi + arm, azimuth)

            # Put everything in the output event
            # The assummed probability assumes that phi is measured
            # exactly, all the uncertainty comes from the error in
            # psichi (through the ARM)
            yield EmCDSEventInSCFrame(measured_energy_keV, phi, psichi.lon.rad, psichi.lat.rad)


    def _effective_area_cm2(self, photons: PolDirESCPhotonList) -> Iterable[float]:
        """

        """
        return [a.to_value(u.cm*u.cm) for a in self._effective_area(photons)]


class IdealComptonIRF(UnpolarizedIdealComptonIRF, FarFieldSpectralPolarizedInstrumentResponseFunctionInterface):

    def _az_prob(self, photon, phi, az):
        pa = photon.polarization_angle_rad_stereo
        return KleinNishinaAzimuthalScatteringAngleDist(photon.energy, phi).pdf((az - pa) % (2 * np.pi))

    def _random_az(self, photon, phi):
        pa = photon.polarization_angle_rad_stereo
        return KleinNishinaAzimuthalScatteringAngleDist(photon.energy, phi).rvs() + pa

class RandomEventDataFromLineInSCFrame(EmCDSEventDataInSCFrameInterface):

    def __init__(self,
                 irf:FarFieldInstrumentResponseFunctionInterface,
                 flux:Quantity,
                 duration:Quantity,
                 energy:Quantity,
                 direction:SkyCoord,
                 polarized_irf:FarFieldInstrumentResponseFunctionInterface,
                 polarization_degree:float = None,
                 polarization_angle:Union[Angle, Quantity] = None,
                 polarization_convention:PolarizationConvention = None):
        """

        Parameters
        ----------
        irf: Must handle PhotonWithDirectionAndEnergyInSCFrameInterface
        flux: Source flux in unit of 1/area/time
        duration: Integration time
        energy: Source energy (a line)
        direction: Source direction (in SC coordinates)
        polarized_irf: Must handle PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface
        polarization_degree
        polarization_angle
        polarization_convention
        """

        unpolarized_irf = irf

        self.event_type = unpolarized_irf.event_type

        flux_cm2_s = flux.to_value(1/u.cm/u.cm/u.s)
        duration_s = duration.to_value(u.s)

        energy_keV = energy.to_value(u.keV)
        direction = direction.transform_to('spacecraftframe')
        source_direction_lon_rad = direction.lon.rad
        source_direction_lat_rad = direction.lat.rad

        unpolarized_photon = PhotonWithDirectionAndEnergyInSCFrame(source_direction_lon_rad,
                                                                         source_direction_lat_rad,
                                                                         energy_keV)

        unpolarized_expected_counts = irf.effective_area_cm2(unpolarized_photon) * flux_cm2_s * duration_s

        if polarization_degree is None:
            polarization_degree = 0

        if polarization_degree < 0 or polarization_degree > 1:
            raise ValueError(f"polarization_degree must lie between 0 and 1. Got {polarization_degree}")

        if polarization_degree == 0:
            polarized_irf = None
            polarized_expected_counts = 0
            polarized_photon = None
        else:

            polarized_irf = polarized_irf

            if polarized_irf.event_type is not unpolarized_irf.event_type:
                raise TypeError(f"Both IRF need to have the same event type. Got {unpolarized_irf.event_type} and {polarized_irf.event_type}")

            polarization_angle_rad = PolarizationAngle(polarization_angle, direction, polarization_convention).transform_to('stereographic').angle.rad

            polarized_photon =  PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention(source_direction_lon_rad,
                                                                                                            source_direction_lat_rad,
                                                                                                            energy_keV,
                                                                                                            polarization_angle_rad)

            unpolarized_expected_counts *= (1 - polarization_degree)
            polarized_expected_counts = polarization_degree * polarized_irf.effective_area_cm2(polarized_photon) * flux_cm2_s * duration_s

        unpolarized_counts = poisson(unpolarized_expected_counts).rvs()
        polarized_counts = poisson(polarized_expected_counts).rvs()

        self._events = []

        unpolarized_photons = unpolarized_irf.photon_list_type.from_photon(unpolarized_photon, repeat = unpolarized_counts)
        unpolarized_events = iter(unpolarized_irf.random_events(unpolarized_photons))

        polarized_events = None
        if polarized_counts > 0:
            polarized_photons = unpolarized_irf.photon_list_type.from_photon(polarized_photon,repeat=polarized_counts)
            polarized_events = iter(polarized_irf.random_events(polarized_photons))

        nthrown_unpolarized = 0
        nthrown_polarized = 0

        while nthrown_unpolarized < unpolarized_counts or nthrown_polarized < polarized_counts:

            if np.random.uniform() < polarization_degree:
                # Polarized component
                if nthrown_polarized < polarized_counts:
                    self._events.append(next(polarized_events))
                    nthrown_polarized += 1
            else:
                # Unpolarized component
                if nthrown_unpolarized < unpolarized_counts:
                    self._events.append(next(unpolarized_events))
                    nthrown_unpolarized += 1

    def __iter__(self) -> Iterator[EmCDSEventInSCFrameInterface]:
        """
        Return one Event at a time
        """
        yield from self._events

    @property
    def nevents(self) -> int:
        return len(self._events)

class ExpectationFromLineInSCFrame(ExpectationDensityInterface):

    def __init__(self,
                 data:EmCDSEventDataInSCFrameInterface,
                 irf:FarFieldInstrumentResponseFunctionInterface,
                 flux:Quantity,
                 duration:Quantity,
                 energy:Quantity,
                 direction:SkyCoord,
                 polarized_irf:FarFieldInstrumentResponseFunctionInterface,
                 polarization_degree:float = None,
                 polarization_angle:Union[Angle, Quantity] = None,
                 polarization_convention:PolarizationConvention = None):

        self._unpolarized_irf = irf
        self._polarized_irf = polarized_irf

        self._duration_s = duration.to_value(u.s)
        self._data = data

        self._flux_cm2_s = None
        self._energy_keV = None
        self._direction = None
        self._source_direction_lon_rad = None
        self._source_direction_lat_rad = None
        self._polarization_degree = None
        self._polarization_angle_rad = None
        self._polarization_convention = None
        self._unpolarized_photon = None
        self._polarized_photon = None
        self.set_model(flux = flux,
                       energy= energy,
                       direction=direction,
                       polarization_degree=polarization_degree,
                       polarization_angle=polarization_angle,
                       polarization_convention = polarization_convention)

        # Cache
        self._cached_energy_keV = None
        self._cached_direction = None
        self._cached_pol_angle_rad = None
        self._cached_pol_degree = None
        self._cached_diff_aeff = None # Per flux unit
        self._cached_event_probability = None
        self._cached_event_probability_unpolarized = None
        self._cached_event_probability_polarized = None

    def set_model(self,
                  flux:Quantity = None,
                  energy:Quantity = None,
                  direction:SkyCoord = None,
                  polarization_degree: float = None,
                  polarization_angle: Union[Angle, Quantity] = None,
                  polarization_convention: PolarizationConvention = None
                  ):
        """
        Parameters not set default to current values
        """

        if flux is not None:
            self._flux_cm2_s = flux.to_value(1 / u.cm / u.cm / u.s)

        if energy is not None:
            self._energy_keV = energy.to_value(u.keV)

        if direction is not None:
            direction = direction.transform_to('spacecraftframe')
            self._direction = direction
            self._source_direction_lon_rad = direction.lon.rad
            self._source_direction_lat_rad = direction.lat.rad

        if polarization_degree is not None:
            self._polarization_degree = polarization_degree

        if self._polarization_degree is None:
            self._polarization_degree = 0

        if self._polarization_degree < 0 or self._polarization_degree > 1:
            raise ValueError(f"polarization_degree must lie between 0 and 1. Got {self._polarization_degree}")

        if self._polarization_degree > 0:

            if self._polarized_irf is None:
                raise ValueError("Polarization degree >0 but polarized IRF is None")

            if polarization_convention is not None:
                self._polarization_convention = polarization_convention

            if polarization_angle is not None:
                self._polarization_angle_rad = PolarizationAngle(polarization_angle, self._direction,
                                                                 self._polarization_convention).transform_to('stereographic').angle.rad

        self._unpolarized_photon = PhotonWithDirectionAndEnergyInSCFrame(self._source_direction_lon_rad,
                                                                         self._source_direction_lat_rad,
                                                                         self._energy_keV)

        self._polarized_photon = PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention(
                                            self._source_direction_lon_rad,
                                            self._source_direction_lat_rad,
                                            self._energy_keV,
                                            self._polarization_angle_rad)

    def _update_cache(self):

        if (self._cached_energy_keV is None
            or self._energy_keV != self._cached_energy_keV
            or self._direction != self._cached_direction
            or self._polarization_angle_rad != self._cached_pol_angle_rad
            or self._polarization_degree != self._cached_pol_degree):
            #Either it's the first time or the energy changed

            unpolarized_diff_aeff = (1 - self._polarization_degree) * self._unpolarized_irf.effective_area_cm2(self._unpolarized_photon)

            if (self._cached_event_probability_unpolarized is None
                    or self._energy_keV != self._cached_energy_keV
                    or self._direction != self._cached_direction):
                # Energy or direction can affect the unpolarized
                # response, but not PA nor PD
                unpolarized_photons = self._unpolarized_irf.photon_list_type.from_photon(self._unpolarized_photon, repeat = self._data.nevents)
                self._cached_event_probability_unpolarized = np.fromiter(self._unpolarized_irf.event_probability(unpolarized_photons, self._data),dtype=float)

            if self._polarization_degree > 0:

                polarized_diff_aeff = self._polarization_degree * self._polarized_irf.effective_area_cm2(self._polarized_photon)

                self._cached_diff_aeff = unpolarized_diff_aeff + polarized_diff_aeff

                if (self._cached_event_probability_polarized is None
                        or self._energy_keV != self._cached_energy_keV
                        or self._direction != self._cached_direction
                        or self._polarization_angle_rad != self._cached_pol_angle_rad):
                    # Energy, direction or PA can affect the
                    # unpolarized response, but not PD
                    polarized_photons = self._polarized_irf.photon_list_type.from_photon(self._polarized_photon, repeat = self._data.nevents)
                    self._cached_event_probability_polarized = np.fromiter(self._polarized_irf.event_probability(polarized_photons, self._data), dtype=float)

                self._cached_event_probability = ( 1 - self._polarization_degree) * self._cached_event_probability_unpolarized + self._polarization_degree * self._cached_event_probability_polarized

            else:

                self._cached_diff_aeff = unpolarized_diff_aeff

                self._cached_event_probability = self._cached_event_probability_unpolarized

            self._cached_energy_keV = self._energy_keV
            self._cached_direction = self._direction
            self._cached_pol_angle_rad = self._polarization_angle_rad
            self._cached_pol_degree = self._polarization_degree

    def expected_counts(self) -> float:

        self._update_cache()

        return self._cached_diff_aeff * (self._flux_cm2_s * self._duration_s)

    def event_probability(self) -> Iterable[float]:

        self._update_cache()

        return self._cached_event_probability
