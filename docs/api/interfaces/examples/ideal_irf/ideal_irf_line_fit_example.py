import logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

import cProfile
import time

import numpy as np
from astropy.coordinates import SkyCoord, Angle

from astropy import units as u
from cosipy.polarization import StereographicConvention
from cosipy.response.ideal_response import IdealComptonIRF, UnpolarizedIdealComptonIRF, ExpectationFromLineInSCFrame, RandomEventDataFromLineInSCFrame
from cosipy.response.relative_coordinates import RelativeCDSCoordinates
from cosipy.statistics import UnbinnedLikelihood
from histpy import Histogram, Axis, HealpixAxis
from matplotlib import pyplot as plt
from scoords import SpacecraftFrame

from mhealpy import HealpixMap
from tqdm import tqdm

#plt.ion()

# ==== Initial setup ====
# Simulated source parameters
source_energy = 500 * u.keV
source_direction = SkyCoord(lon = 0, lat = 60, unit = 'deg', frame = SpacecraftFrame())
source_flux = 1 / u.cm / u.cm / u.s
source_pd = .7
source_pa = 80 * u.deg
pol_convention = StereographicConvention()

# The integrated time of the observation. Increase it to get more statistics
# which is good for visualizing the data, but that will maka the analysis slower
duration = 10*u.s

# Instrument Response Function (IRF) definitions
# The "unpolarized" response returns an average over all
# polarization angles.
irf_pol = IdealComptonIRF.cosi_like()
irf_unpol = UnpolarizedIdealComptonIRF.cosi_like()

# Simulate data sampling from the IRF itself
# This simulated a monochromatic source at a fixed direction in the SC coordinate frame

# profile = cProfile.Profile()
# profile.enable()
# tstart = time.perf_counter()
logger.info("Simulating data...")
data = RandomEventDataFromLineInSCFrame(irf = irf_unpol,
                                        flux = source_flux,
                                        duration = duration,
                                        energy=source_energy,
                                        direction =  source_direction,
                                        polarized_irf= irf_pol,
                                        polarization_degree=source_pd,
                                        polarization_angle=source_pa,
                                        polarization_convention=pol_convention)

# Get the measured energy (Em) and the Compton Data Space (CDS) (CDS = Phi and PsiChi)
measured_energy = data.energy
phi = data.scattering_angle
psichi = data.scattered_direction_sc

logger.info(f"Got {data.nevents} events.")

# ======= Data visualization ======

fig,ax = plt.subplots(2, 3, figsize = [18,8])

# This is a visualization of the Compton cone. Instead of drawing it in 3D space,
# we'll use color to represent the scattering angle Phi, which is usually the z-axis a Compton cone plot.
# The location of the source is marked by an X, and the direction of each scattered photon (PsiChi) is represented
# with a dot
ax[0,0].set_axis_off() # Replace corner plot with axis suitable for spherical data
sph_ax = fig.add_subplot(2,3,1, projection = 'mollview')

sc = sph_ax.scatter(psichi.lon.deg, psichi.lat.deg, transform = sph_ax.get_transform('world'), c = phi.to_value('deg'), cmap = 'inferno',
                s = 2, vmin = 0, vmax = 180)
sph_ax.scatter(source_direction.lon.deg, source_direction.lat.deg, transform = sph_ax.get_transform('world'), marker = 'x', s = 100, c = 'red')
fig.colorbar(sc, orientation="horizontal", fraction = .05, label = "phi [deg]")

sph_ax.set_title("Compton Data Space")

# While the data live in this complex 4-D space (Em + CDS) it is useful to make some cuts and projections
# to visualize it. For this we use the following coordinates, which are relative to hypothetical source
# (or, in this case, a known source, since we simulated it)
# Epsilon: fractional difference in energy which respect to the energy of the source
# Phi_geometric: angular distance between the source location and PsiChi
# Theta_ARM = the difference between Phi (which is computed exclusively from kinematics) and Phi_geometric
# Zeta: the azimuthal scattering direction, computed from PsiChi once a particular source direction is assumed.
#       The zeta=0 direction is arbitrary, and is defined by the polarization convention.
eps = ((measured_energy - source_energy) / source_energy).to_value('')
phi_geom,zeta = RelativeCDSCoordinates(source_direction, pol_convention).to_relative(psichi)
theta_arm = phi_geom - phi

rel_binned_data = Histogram([Axis(np.linspace(-1,1.1,200), scale = 'linear', label='eps'),
                             Axis(np.linspace(0, 180, 180)*u.deg, scale='linear', label='phi'),
                             Axis(np.linspace(-180, 180, 180)*u.deg, scale='linear', label='arm'),
                             Axis(np.linspace(-180, 180, 180) * u.deg, scale='linear', label='az')])

rel_binned_data.fill(eps, phi, theta_arm, zeta)

rel_binned_data.slice[{'phi':slice(30,120)}].project('az').rebin(5).plot(ax[1,0],errorbars = True)
ax[1,0].set_title("Azimuthal Scattering Angle Distribution (ASAD)")

rel_binned_data.project(['arm','phi']).rebin(3,5).plot(ax[0,1])
ax[0,1].set_title('Compton cone "wall"')

rel_binned_data.project('phi').rebin(5).plot(ax[0,2],errorbars = True)
ax[0,2].set_title("Polar Scattering Angle")

rel_binned_data.project('arm').rebin(3).plot(ax[1,1],errorbars = True)
ax[1,1].set_title("Angular Resolution Measure (ARM)")

rel_binned_data.project('eps').plot(ax[1,2],errorbars = True)
ax[1,2].set_title("Energy dispersion")

fig.subplots_adjust(left=.05, right=.95, top=.95, bottom=.1, wspace=0.2, hspace=0.4)

plt.show()

# ===== Likelihood setup =====

# In order to compute the likelihood we need to know how many counts we expect and,
# if they are detected, what is the probability density of having obtained a
# specific Em+CDS set of parameters. All of this is computed from the IRF's
# effective area and event probability density functions (PDFs).
# Since we used exactly the same effective area and PDFs to simulated our event,
# then we should get the "perfect" result. There will be statistical fluctuations
# resulting in a statistical error, but no systematic error.

expectation = ExpectationFromLineInSCFrame(data,
                                           irf=irf_unpol,
                                           flux=source_flux,
                                           duration=duration,
                                           energy=source_energy,
                                           direction=source_direction,
                                           polarized_irf=irf_pol,
                                           polarization_degree=source_pd,
                                           polarization_angle=source_pa,
                                           polarization_convention=pol_convention)

likelihood = UnbinnedLikelihood(expectation)

# ==== Fits ====

# We'll use a brute-force maximum-likelihood estimation technique. That is, we'll compute
# likelihood as a function of all free parameters, get the combination that maximizes the likelihood,
# and use Wilks theorem to obtain an estimate of the errors.

# We'll only free one parameter at a time, and set all others to known values.
# The flux will always be a "nuisance" parameter

fit_energy = False
fit_direction = False
direction_nside = 128 # Decrease/increase it to get a better/worse TS map. It'll be faster/slower
fit_pa_pd = True

# ==== Free the source energy ====
if fit_energy:
    # Set everything to the injection values
    expectation.set_model(flux=source_flux,
                          energy=source_energy,
                          direction=source_direction,
                          polarization_degree=source_pd,
                          polarization_angle=source_pa)

    # Compute the likelihood on a grid
    loglike = Histogram([Axis(np.linspace(.8, 1.2, 11)/u.cm/u.cm/u.s, label = 'flux'),
                            Axis(np.linspace(499, 501, 10)*u.keV, label = 'Ei')])

    for j, ei_j in tqdm(list(enumerate(loglike.axes['Ei'].centers)), desc="Likelihood (free energy)"):
        for i,flux_i in enumerate(loglike.axes['flux'].centers):

            expectation.set_model(flux = flux_i, energy = ei_j)
            loglike[i,j] = likelihood.get_log_like()

    # Use Wilks theorem to get a 90% confidence interval
    ts = 2 * (loglike - np.max(loglike))
    ax,_ = ts.plot(vmin = -4.61)
    ax.scatter(source_flux.to_value(loglike.axes['flux'].unit), source_energy.to_value(loglike.axes['Ei'].unit),
               color='red')
    ax.get_figure().axes[-1].set_ylabel("TS")

    plt.show()


# ==== Free the source direction ====
if fit_direction:
    # Set everything to the injection values
    expectation.set_model(flux=source_flux,
                          energy=source_energy,
                          direction=source_direction,
                          polarization_degree=source_pd,
                          polarization_angle=source_pa)

    loglike = Histogram([Axis(np.linspace(.8, 1.2, 11)/u.cm/u.cm/u.s, label = 'flux'),
                         HealpixAxis(nside = direction_nside, label = 'direction', coordsys=SpacecraftFrame())])

    loglike[:] = np.nan

    sample_pixels = loglike.axes['direction'].query_disc(source_direction.cartesian.xyz, np.deg2rad(3))
    for j, pix in tqdm(list(enumerate(sample_pixels)), desc="Likelihood (direction)"):

        coord_pix = loglike.axes['direction'].pix2skycoord(pix)

        for i,flux_i in enumerate(loglike.axes['flux'].centers):

            expectation.set_model(flux = flux_i, direction = coord_pix)

            loglike[i,pix] = likelihood.get_log_like()


    fig,ax = plt.subplots(1,2, figsize = [10,4])

    ax[0].set_axis_off()  # Replace corner plot with axis suitable for spherical data
    sph_ax = fig.add_subplot(1, 2, 1, projection='cartview', latra = source_direction.lat.deg + [-3,3], lonra = source_direction.lon.deg + np.asarray([-3,3])/np.cos(source_direction.lat.rad))
    sph_ax.coords[0].set_ticks_visible(True)
    sph_ax.coords[1].set_ticks_visible(True)
    sph_ax.coords[0].set_ticklabel_visible(True)
    sph_ax.coords[1].set_ticklabel_visible(True)

    direction_profile_loglike = HealpixMap(np.nanmax(loglike, axis = 0))
    ts_direction = 2*(direction_profile_loglike - np.nanmax(direction_profile_loglike))
    ts_direction.plot(sph_ax, vmin = -4.61)
    sph_ax.set_title("Location TS map")
    sph_ax.get_figure().axes[-1].set_xlabel("TS")
    sph_ax.scatter(source_direction.lon.deg, source_direction.lat.deg, transform=sph_ax.get_transform('world'),
                   marker='x', s=100, c='red')

    flux_prof_loglike = Histogram(loglike.axes['flux'], contents = np.nanmax(loglike, axis = 1))
    ts_flux = 2*(flux_prof_loglike - np.nanmin(flux_prof_loglike))
    ts_flux.plot(ax[1])
    ax[1].axvline(source_flux.to_value(loglike.axes['flux'].unit), color = 'red', ls = ':')
    ax[1].set_title("Flux TS profile")

    plt.show()

# ==== Free PD and PA ====
if fit_pa_pd:
    # Set everything to the injection values
    expectation.set_model(flux=source_flux,
                          energy=source_energy,
                          direction=source_direction,
                          polarization_degree=source_pd,
                          polarization_angle=source_pa)

    loglike = Histogram([Axis(np.linspace(.8, 1.2, 11) / u.cm / u.cm / u.s, label='flux'),
                         Axis(np.linspace(40,120,10)*u.deg, label='PA'),
                         Axis(np.linspace(0, 1, 11), label='PD'),
                         ])

    for j, pa_j in tqdm(list(enumerate(loglike.axes['PA'].centers)), desc="Likelihood (polarization)"):
        for k, pd_k in enumerate(loglike.axes['PD'].centers):
            for i,flux_i in enumerate(loglike.axes['flux'].centers):

                expectation.set_model(flux = flux_i,
                                      polarization_degree = pd_k,
                                      polarization_angle = pa_j)

                loglike[i,j,k] = likelihood.get_log_like()

    fig,ax = plt.subplots(1,2, figsize = [10,4])

    flux_prof_loglike = Histogram(loglike.axes['flux'], contents = np.nanmax(loglike, axis = (1,2)))
    ts_flux = 2 * (flux_prof_loglike - np.nanmin(flux_prof_loglike))
    ts_flux.plot(ax[0])
    ax[0].axvline(source_flux.to_value(loglike.axes['flux'].unit), color='red', ls=':')
    ax[0].set_ylabel("TS")
    ax[0].set_title("Flux TS profile")

    pol_prof_loglike = Histogram([loglike.axes['PA'], loglike.axes['PD']], contents=np.nanmax(loglike, axis=0))

    ts_pol = 2 * (pol_prof_loglike - np.nanmax(pol_prof_loglike))
    ts_pol.plot(ax[1], vmin=-4.61)
    ax[1].scatter(source_pa.to_value(u.deg), source_pd, color='red')
    ax[1].get_figure().axes[-1].set_ylabel("TS")
    ax[1].set_title("PA-PD TS profile")

    plt.show()

plt.show()