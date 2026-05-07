#!/usr/bin/env python
# coding: utf-8

# # Spectral fitting example (Crab)

# **To run this, you need the following files, which can be downloaded using the first few cells of this notebook:**
# - orientation file (20280301_3_month_with_orbital_info.fits)
# - binned data (crab_bkg_binned_data.hdf5, crab_binned_data.hdf5, & bkg_binned_data.hdf5)
# - detector response (SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5)
#
# **The binned data are simulations of the Crab Nebula and albedo photon background produced using the COSI SMEX mass model. The detector response needs to be unzipped before running the notebook.**

# This notebook fits the spectrum of a Crab simulated using MEGAlib and combined with background.
#
# [3ML](https://threeml.readthedocs.io/) is a high-level interface that allows multiple datasets from different instruments to be used coherently to fit the parameters of source model. A source model typically consists of a list of sources with parametrized spectral shapes, sky locations and, for extended sources, shape. Polarization is also possible. A "coherent" analysis, in this context, means that the source model parameters are fitted using all available datasets simultanously, rather than performing individual fits and finding a well-suited common model a posteriori.
#
# In order for a dataset to be included in 3ML, each instrument needs to provide a "plugin". Each plugin is responsible for reading the data, convolving the source model (provided by 3ML) with the instrument response, and returning a likelihood. In our case, we'll compute a binned Poisson likelihood:
#
# $$
# \log \mathcal{L}(\mathbf{x}) = \sum_i \log \frac{\lambda_i(\mathbf{x})^{d_i} \exp (-\lambda_i)}{d_i!}
# $$
#
# where $d_i$ are the counts on each bin and $\lambda_i$ are the expected counts given a source model with parameters $\mathbf{x}$.
#
# In this example, we will fit a single point source with a known location. We'll assume the background is known and fixed up to a scaling factor. Finally, we will fit a Band function:
#
# $$
# f(x) = K \begin{cases} \left(\frac{x}{E_{piv}}\right)^{\alpha} \exp \left(-\frac{(2+\alpha)
#        * x}{x_{p}}\right) & x \leq (\alpha-\beta) \frac{x_{p}}{(\alpha+2)} \\ \left(\frac{x}{E_{piv}}\right)^{\beta}
#        * \exp (\beta-\alpha)\left[\frac{(\alpha-\beta) x_{p}}{E_{piv}(2+\alpha)}\right]^{\alpha-\beta}
#        * &x>(\alpha-\beta) \frac{x_{p}}{(\alpha+2)} \end{cases}
# $$
#
# where $K$ (normalization), $\alpha$ & $\beta$ (spectral indeces), and $x_p$ (peak energy) are the free parameters, while $E_{piv}$ is the pivot energy which is fixed (and arbitrary).
#
# Considering these assumptions:
#
# $$
# \lambda_i(\mathbf{x}) = B*b_i + s_i(\mathbf{x})
# $$
#
# where $B*b_i$ are the estimated counts due to background in each bin with $B$ the amplitude and $b_i$ the shape of the background, and $s_i$ are the corresponding expected counts from the source, the goal is then to find the values of $\mathbf{x} = [K, \alpha, \beta, x_p]$ and $B$ that maximize $\mathcal{L}$. These are the best estimations of the parameters.
#
# The final module needs to also fit the time-dependent background, handle multiple point-like and extended sources, as well as all the spectral models supported by 3ML. Eventually, it will also fit the polarization angle. However, this simple example already contains all the necessary pieces to do a fit.

# In[1]:


from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.util import fetch_wasabi_file

from cosipy.statistics import PoissonLikelihood
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse
from cosipy.data_io import EmCDSBinnedData

import sys

import astropy.units as u

import numpy as np
import matplotlib.pyplot as plt

from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter, Powerlaw

from pathlib import Path



def main():

    single_bkg_fit = True

    # ## Download and read in binned data

    # Define the path to the directory containing the data, detector response, orientation file, and yaml files if they have already been downloaded, or the directory to download the files into

    data_path = Path("") # /path/to/files. Current dir by default


    # Download the orientation file


    # In[ ]:

    sc_orientation_path = data_path / "DC3_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.ori"
    fetch_wasabi_file('COSI-SMEX/DC3/Data/Orientation/DC3_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.ori',
                      output=sc_orientation_path, checksum = 'e5e71e3528e39b855b0e4f74a1a2eebe')

    # Download the binned Crab data

    # In[7]:

    crab_data_path = data_path / "crab_standard_3months_binned_data_filtered_with_SAAcut.fits.gz.hdf5"
    fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/crab_spectral_fit_galactic_frame/crab_standard_3months_binned_data_filtered_with_SAAcut.fits.gz.hdf5',
                      output=crab_data_path, checksum = '405862396dea2be79d7892d6d5bb50d8')

    bkg_components = {"PrimaryProtons":{'filename':'PrimaryProtons_WithDetCstbinned_data_filtered_with_SAAcut.hdf5', 'checksum':'7597f04210e59340a0888c66fc5cbc63'},
                      "PrimaryAlphas": {'filename': 'PrimaryAlphas_WithDetCstbinned_data_filtered_with_SAAcut.hdf5', 'checksum': '76a68da730622851b8e1c749248c3b40'},
                      "AlbedoPhotons": {'filename': 'AlbedoPhotons_WithDetCstbinned_data_filtered_with_SAAcut.hdf5', 'checksum': '76c58361d2c9b43b66ef2e41c18939c4'},
                      "AlbedoNeutrons": {'filename': 'AlbedoNeutrons_WithDetCstbinned_data_filtered_with_SAAcut.hdf5', 'checksum': '8f3cb418c637b839665a4fcbd000d2eb'},
                      "CosmicPhotons": {'filename': 'CosmicPhotons_3months_binned_data_filtered_with_SAAcut.hdf5', 'checksum': '93c4619b383572d318328e6380e35a70'},
                      "CosmicDiffuse": {'filename': 'GalTotal_SA100_F98_3months_binned_data_filtered_with_SAAcut.hdf5', 'checksum': 'd0415d4d04b040af47f23f5d08cb7d64'},
                      "SecondaryPositrons": {'filename': 'SecondaryPositrons_3months_binned_data_filtered_with_SAAcut.hdf5', 'checksum': '5fec2212dcdbb4c43c3ac02f02524f68'},
                      "SecondaryProtons": {'filename': 'SecondaryProtons_WithDetCstbinned_data_filtered_with_SAAcut.fits.gz.hdf5', 'checksum': '78aefa46707c98563294a898a62845c1'},
                      "SAAprotons": {'filename': 'SAA_3months_unbinned_data_filtered_with_SAAcut_statreduced_akaHEPD01result.hdf5', 'checksum': 'fc69fbbfd94cd595f57a8b11fc721169'},
                      }

   # Download the binned background data
    for bkg in bkg_components.values():
        wasabi_path = 'COSI-SMEX/cosipy_tutorials/crab_spectral_fit_galactic_frame/'+bkg['filename']
        fetch_wasabi_file(wasabi_path, output=data_path/bkg['filename'], checksum = bkg['checksum'])

    # Download the response file
    dr_path = data_path / "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5"
    fetch_wasabi_file('COSI-SMEX/develop/Data/Responses/ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5',
                       output=str(dr_path), checksum = '7121f094be50e7bfe9b31e53015b0e85')


    # Read in the spacecraft orientation file

    # In[4]:


    sc_orientation = SpacecraftHistory.open(sc_orientation_path)


    # Create BinnedData objects for the Crab only, Crab+background, and background only. The Crab only simulation is not used for the spectral fit, but can be used to compare the fitted spectrum to the source simulation

    # In[5]:


    crab = BinnedData(data_path / "crab.yaml")
    crab.load_binned_data_from_hdf5(binned_data=crab_data_path)

    for bkg in bkg_components.values():
        binned_data = BinnedData(data_path / "background.yaml")
        binned_data.load_binned_data_from_hdf5(binned_data=data_path/bkg['filename'])
        bkg['dist'] = binned_data.binned_data.project('Em', 'Phi', 'PsiChi')

    # Load binned .hdf5 files

    # In[6]:


    # Define the path to the detector response
    # ## Perform spectral fit

    # ============ Interfaces ==============

    # Set background parameter, which is used to fit the amplitude of the background, and instantiate the COSI 3ML plugin

    # In[8]:
    total_bkg = None
    for bkg in bkg_components.values():
        if total_bkg is None:
            total_bkg = bkg['dist']
        else:
            total_bkg = total_bkg + bkg['dist'] # Issues with in-place operations for sparse contents

    if single_bkg_fit:
        bkg_dist = {"total_bkg":total_bkg}
    else:
        bkg_dist = {l: b['dist'] for l, b in bkg_components.items()}

    # Workaround to avoid inf values. Out bkg should be smooth, but currently it's not.
    # Reproduces results before refactoring. It's not _exactly_ the same, since this fudge value was 1e-12, and
    # it was added to the expectation, not the normalized bkg
    for bckfile in bkg_dist.keys() :
        bkg_dist[bckfile] += sys.float_info.min

    #combine the data + the bck like we would get for real data
    data = EmCDSBinnedData(crab.binned_data.project('Em', 'Phi', 'PsiChi') + total_bkg)
    bkg = FreeNormBinnedBackground(bkg_dist,
                                   sc_history=sc_orientation,
                                   copy = False)

    dr = FullDetectorResponse.open(dr_path)
    instrument_response = BinnedInstrumentResponse(dr, data)

    # Currently using the same NnuLambda, Ei and Pol axes as the underlying FullDetectorResponse,
    # matching the behavior of v0.3. This is all the current BinnedInstrumentResponse can do.
    # In principle, this can be decoupled, and a BinnedInstrumentResponseInterface implementation
    # can provide the response for an arbitrary directions, Ei and Pol values.
    psr = BinnedThreeMLPointSourceResponse(data = data,
                                           instrument_response = instrument_response,
                                           sc_history=sc_orientation,
                                           energy_axis = dr.axes['Ei'],
                                           polarization_axis = dr.axes['Pol'] if 'Pol' in dr.axes.labels else None,
                                           nside = 2*data.axes['PsiChi'].nside)

    ##====


    response = BinnedThreeMLModelFolding(data = data, point_source_response = psr)

    like_fun = PoissonLikelihood(data, response, bkg)

    cosi = ThreeMLPluginInterface('cosi',
                                  like_fun,
                                  response,
                                  bkg)

    # Nuisance parameter guess, bounds, etc.
    for bkg_label in bkg_dist.keys():
        cosi.bkg_parameter[bkg_label] = Parameter(bkg_label,  # background parameter
                                          1,  # initial value of parameter
                                          min_value=0,  # minimum value of parameter
                                          max_value= 100 if single_bkg_fit else 20,  # maximum value of parameter
                                          delta=0.05,  # initial step used by fitting engine
                                          unit = u.Hz
                                          )

    # ======== Interfaces end ==========

    # Define a point source at the known location with a Band function spectrum and add it to the model. The initial values of the Band function parameters are set to the true values used to simulate the source


    # In[9]:

    l = 184.56
    b = -5.78

    alpha = -1.99
    beta = -2.32
    E0 = 531. * (alpha - beta) * u.keV
    xp = E0 * (alpha + 2) / (alpha - beta)
    piv = 500. * u.keV
    K = 3.07e-5 / u.cm / u.cm / u.s / u.keV

    spectrum = Band()

    spectrum.alpha.min_value = -2.14
    spectrum.alpha.max_value = 3.0
    spectrum.beta.min_value = -5.0
    spectrum.beta.max_value = -2.15
    spectrum.xp.min_value = 1.0
    spectrum.K.min_value = 1e-10

    spectrum.alpha.value = alpha
    spectrum.beta.value = beta
    spectrum.xp.value = xp.value
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value

    spectrum.xp.unit = xp.unit
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    spectrum.alpha.delta = 0.01
    spectrum.beta.delta = 0.01

    source = PointSource("source",  # Name of source (arbitrary, but needs to be unique)
                         l=l,  # Longitude (deg)
                         b=b,  # Latitude (deg)
                         spectral_shape=spectrum)  # Spectral model

    model = Model(
        source)  # Model with single source. If we had multiple sources, we would do Model(source1, source2, ...)

    # Optional: if you want to call get_log_like manually, then you also need to set the model manually
    # 3ML does this internally during the fit though
    cosi.set_model(model)


    # Gather all plugins and combine with the model in a JointLikelihood object, then perform maximum likelihood fit

    # In[10]:


    plugins = DataList(cosi) # If we had multiple instruments, we would do e.g. DataList(cosi, lat, hawc, ...)

    like = JointLikelihood(model, plugins, verbose = False)

    like.fit()


    # ## Error propagation and plotting (Band function)

    # Define Band function spectrum injected into MEGAlib

    # In[11]:

    ## Injected

    l = 184.56
    b = -5.78

    alpha_inj = -1.99
    beta_inj = -2.32
    E0_inj = 531. * (alpha_inj - beta_inj) * u.keV
    xp_inj = E0_inj * (alpha_inj + 2) / (alpha_inj - beta_inj)
    piv_inj = 100. * u.keV
    K_inj = 7.56e-4 / u.cm / u.cm / u.s / u.keV

    spectrum_inj = Band()

    spectrum_inj.alpha.min_value = -2.14
    spectrum_inj.alpha.max_value = 3.0
    spectrum_inj.beta.min_value = -5.0
    spectrum_inj.beta.max_value = -2.15
    spectrum_inj.xp.min_value = 1.0
    spectrum_inj.K.min_value = 1e-10

    spectrum_inj.alpha.value = alpha_inj
    spectrum_inj.beta.value = beta_inj
    spectrum_inj.xp.value = xp_inj.value
    spectrum_inj.K.value = K_inj.value
    spectrum_inj.piv.value = piv_inj.value

    spectrum_inj.xp.unit = xp_inj.unit
    spectrum_inj.K.unit = K_inj.unit
    spectrum_inj.piv.unit = piv_inj.unit

    # Expectation for injected source
    source_inj = PointSource("source",  # Name of source (arbitrary, but needs to be unique)
                             l=l,  # Longitude (deg)
                             b=b,  # Latitude (deg)
                             spectral_shape=spectrum_inj)  # Spectral model

    psr.set_source(source_inj)
    expectation_inj = psr.expectation(copy=True)


    # The summary of the results above tell you the optimal values of the parameters, as well as the errors. Propogate the errors to the "evaluate_at" method of the spectrum

    # In[12]:


    results = like.results


    print(results.display())

    parameters = {par.name:results.get_variates(par.path)
                  for par in results.optimized_model["source"].parameters.values()
                  if par.free}

    results_err = results.propagate(results.optimized_model["source"].spectrum.main.shape.evaluate_at, **parameters)

    print(results.optimized_model["source"])

    # Evaluate the flux and errors at a range of energies for the fitted and injected spectra, and the simulated source flux

    # In[13]:


    energy = np.geomspace(100*u.keV,10*u.MeV).to_value(u.keV)

    flux_lo = np.zeros_like(energy)
    flux_median = np.zeros_like(energy)
    flux_hi = np.zeros_like(energy)
    flux_inj = np.zeros_like(energy)

    for i, e in enumerate(energy):
        flux = results_err(e)
        flux_median[i] = flux.median
        flux_lo[i], flux_hi[i] = flux.equal_tail_interval(cl=0.68)
        flux_inj[i] = spectrum_inj.evaluate_at(e)

    binned_energy_edges = crab.binned_data.axes['Em'].edges.value
    binned_energy = np.array([])
    bin_sizes = np.array([])

    for i in range(len(binned_energy_edges)-1):
        binned_energy = np.append(binned_energy, (binned_energy_edges[i+1] + binned_energy_edges[i]) / 2)
        bin_sizes = np.append(bin_sizes, binned_energy_edges[i+1] - binned_energy_edges[i])

    expectation = response.expectation(copy = True)


    # Plot the fitted and injected spectra

    # In[14]:


    fig,ax = plt.subplots()

    ax.plot(energy, energy*energy*flux_median, label = "Best fit")
    ax.fill_between(energy, energy*energy*flux_lo, energy*energy*flux_hi, alpha = .5, label = "Best fit (errors)")
    ax.plot(energy, energy*energy*flux_inj, color = 'black', ls = ":", label = "Injected")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ (keV cm$^{-2}$ s$^{-1}$)")

    ax.legend()

    ax.set_ylim(.1,100)

    #plt.show()

    # Plot the fitted spectrum convolved with the response, as well as the simulated source counts

    # In[15]:


    fig,ax = plt.subplots()

    ax.stairs(expectation.project('Em').to_dense(copy=False).contents, binned_energy_edges, color='purple', label = "Best fit convolved with response")
    ax.stairs(expectation_inj.project('Em').to_dense(copy=False).contents, binned_energy_edges, color='blue', label = "Injected spectrum convolved with response")
    ax.errorbar(binned_energy, expectation.project('Em').to_dense(copy=False).contents, yerr=np.sqrt(expectation.project('Em').to_dense(copy=False).contents), color='purple', linewidth=0, elinewidth=1)
    ax.stairs(crab.binned_data.project('Em').to_dense(copy=False).contents, binned_energy_edges, color = 'black', ls = ":", label = "Source counts")
    ax.errorbar(binned_energy, crab.binned_data.project('Em').to_dense(copy=False).contents, yerr=np.sqrt(crab.binned_data.project('Em').to_dense(copy=False).contents), color='black', linewidth=0, elinewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")

    ax.legend()

    #plt.show()


    # Plot the fitted spectrum convolved with the response plus the fitted background, as well as the simulated source+background counts

    # In[16]:

    expectation_bkg = bkg.expectation(copy = True)

    fig,ax = plt.subplots()

    ax.stairs(expectation.project('Em').to_dense(copy=False).contents + expectation_bkg.project('Em').to_dense(copy=False).contents, binned_energy_edges, color='purple', label = "Best fit convolved with response plus background")
    ax.errorbar(binned_energy, expectation.project('Em').to_dense(copy=False).contents+expectation_bkg.project('Em').to_dense(copy=False).contents, yerr=np.sqrt(expectation.project('Em').to_dense(copy=False).contents+expectation_bkg.project('Em').to_dense(copy=False).contents), color='purple', linewidth=0, elinewidth=1)
    ax.stairs(data.data.project('Em').to_dense(copy=False).contents, binned_energy_edges, color = 'black', ls = ":", label = "Total counts")
    ax.errorbar(binned_energy, data.data.project('Em').to_dense(copy=False).contents, yerr=np.sqrt(data.data.project('Em').to_dense(copy=False).contents), color='black', linewidth=0, elinewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")

    ax.legend()

    plt.show()


if __name__ == "__main__":

    import cProfile
    cProfile.run('main()', filename = "prof_interfaces.prof")
    exit()

    main()
