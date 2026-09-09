import cProfile
import itertools
import time

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from cosipy.polarization import StereographicConvention
from cosipy.response.ideal_response import IdealComptonIRF
from astropy import units as u
from cosipy.response.photon_types import PhotonWithDirectionAndEnergyInSCFrame, \
    PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention, PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.response.relative_coordinates import RelativeCDSCoordinates
from histpy import Axis, HealpixAxis, Histogram
from matplotlib import pyplot as plt

from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized

from pathlib import Path

wdir = Path("/Users/imartin5/cosi/scratch/response_relative_coordinates/v4")
plot_dir = wdir/"plots"
plot_dir.mkdir(parents=True, exist_ok=True)

irf = IRFRelativeHistUnpolarized.from_h5(wdir/"ResponseContinuum.area.relative.nonsparse.h5")

### Effective ares

aeff = Histogram([Axis(np.geomspace(100, 10000, 100) * u.keV, scale='log', label='Ei'),
                         HealpixAxis(nside=16, label='NuLambda', coordsys='spacecraftframe')], unit=u.cm * u.cm)

energies_keV = aeff.axes['Ei'].centers

for i in range(aeff.axes['NuLambda'].npix):

    coord = aeff.axes['NuLambda'].pix2skycoord(i)

    photons = PhotonListWithDirectionAndEnergyInSCFrame([coord.lon.rad], [coord.lat.rad], energies_keV)

    aeff[:,i] = Quantity(irf.effective_area_cm2(photons), u.cm * u.cm, copy=False)

ax,_ = aeff.slice[{'NuLambda': 0}].project('Ei').plot()

ax.get_figure().savefig(plot_dir/"aeff_energy.png")

ax,_ = aeff.slice[{'Ei': 25}].project('NuLambda').plot()

ax.get_figure().savefig(plot_dir/"aeff_direction.png")

### Energy dispersion for a source on axis

from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays

# NuLambda pixel 0 is the on-axis (boresight) direction, same convention
# used for the effective-area slice above.
photon_coord = aeff.axes['NuLambda'].pix2skycoord(0)

# Representative Compton scattering angle. The scattered direction is
# placed exactly phi_kin away from the photon direction (Theta = 0), i.e.
# on the response's main kinematic ridge.
phi_kin_deg = 60.0
psichi_coord = photon_coord.directional_offset_by(0 * u.deg, phi_kin_deg * u.deg)

epsilon_grid = np.linspace(-0.25, .25, 3000)

fig, ax = plt.subplots()

for ei_keV in [100, 300, 1000, 3000, 10000]:

    em_keV = ei_keV * (1 + epsilon_grid)

    photons = PhotonListWithDirectionAndEnergyInSCFrame(
        np.full_like(epsilon_grid, photon_coord.lon.rad),
        np.full_like(epsilon_grid, photon_coord.lat.rad),
        np.full_like(epsilon_grid, ei_keV))

    events = EmCDSEventDataInSCFrameFromArrays(
        em_keV,
        np.full_like(epsilon_grid, psichi_coord.lon.rad),
        np.full_like(epsilon_grid, psichi_coord.lat.rad),
        np.full_like(epsilon_grid, np.radians(phi_kin_deg)))

    density = np.fromiter(irf.event_probability(photons, events), dtype=float)

    ax.plot(epsilon_grid, density, label=f'{ei_keV:.0f} keV')

ax.set_xlabel('Epsilon = (Em - Ei) / Ei')
ax.set_ylabel('Event probability density')
ax.set_title(f'Energy dispersion, on-axis source (Phi = {phi_kin_deg:.0f} deg)')
ax.legend()

ax.get_figure().savefig(plot_dir/"energy_dispersion.png")

### ARM (angular resolution measure) for a source on axis

# ARM = Theta = phi_geo - phi_kin: the offset between the geometric
# scattering angle (NuLambda to PsiChi) and the kinematics-derived one.
# Hold Epsilon = 0 (Em = Ei) so energy dispersion doesn't mix in, and
# scan the geometric separation of PsiChi from the photon direction
# around phi_kin.
#
# phi_geo = phi_kin + Theta is a genuine angular separation, so it must
# stay within (0, 180) deg. Outside that range, directional_offset_by()
# still returns a valid point, but at the *folded* separation
# |phi_kin + Theta| (mod the sphere), aliasing it onto a different,
# wrong Theta -- e.g. Theta = -2 * phi_kin folds back onto Theta = 0,
# producing a spurious second peak.
theta_grid_deg = np.linspace(-phi_kin_deg, 180.0 - phi_kin_deg, 3000)

fig, ax = plt.subplots()

for ei_keV in [100, 300, 1000, 3000, 10000]:

    em_keV = np.full_like(theta_grid_deg, ei_keV)

    phi_geo_deg = phi_kin_deg + theta_grid_deg
    arm_psichi_coord = photon_coord.directional_offset_by(0 * u.deg, phi_geo_deg * u.deg)

    photons = PhotonListWithDirectionAndEnergyInSCFrame(
        np.full_like(theta_grid_deg, photon_coord.lon.rad),
        np.full_like(theta_grid_deg, photon_coord.lat.rad),
        np.full_like(theta_grid_deg, ei_keV))

    events = EmCDSEventDataInSCFrameFromArrays(
        em_keV,
        arm_psichi_coord.lon.rad,
        arm_psichi_coord.lat.rad,
        np.full_like(theta_grid_deg, np.radians(phi_kin_deg)))

    density = np.fromiter(irf.event_probability(photons, events), dtype=float)

    ax.plot(theta_grid_deg, density, label=f'{ei_keV:.0f} keV')

ax.set_xlabel('ARM = Theta [deg]')
ax.set_ylabel('Event probability density')
ax.set_title(f'ARM, on-axis source (Phi = {phi_kin_deg:.0f} deg)')
ax.legend()

ax.get_figure().savefig(plot_dir/"arm.png")

exit