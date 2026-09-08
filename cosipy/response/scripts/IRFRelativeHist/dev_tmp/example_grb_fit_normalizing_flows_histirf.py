#!/usr/bin/env python
# coding: utf-8

# # Fitting the GRB Spectrum with the Neural-Network Response and Background Approximation

# ## Introduction
# 
# This notebook provides an overview of the different new classes introduced to allow for a truly unbinned spectral analysis of continuum point sources. The user will mainly interact with:
# 
# 1. `NFResponse` and `UnpolarizedNFFarFieldInstrumentResponseFunction`: Provide the **C-A-RQS + Spherical Harmonics Expansion** approximation of the response.
# 2. `NFBackground` and `FreeNormNFUnbinnedBackground`: Provide the **C-A-RQS + Analytical Rate Model** approximation of the simulated background (currently total DC4).
# 3. `UnbinnedThreeMLPointSourceResponseIRFAdaptive`: Perform the folding of the response with the flux model in a more efficient way.
# 4. `CachedUnbinnedThreeMLModelFolding`: Adds the capability to save and load the cache of `UnbinnedThreeMLPointSourceResponseIRFAdaptive`.
# 
# ### Inner Workings
# 
# For a comprehensive description of the underlying architecture or more detailed comparison plots and benchmarks, please for now refer to my (Pascal J.) thesis, "Development of an Efficient Response Description for the COSI MeV 𝜸-Ray Telescope." Here, I will just provide a very basic explanation of the code structure.
# 
# #### Flow-Based Models
# 
# The building blocks managing the approximations are `NFResponse` and `NFBackground` (both of type `NFBase`). During setup, both require a checkpoint file (e.g., `unpolarized_nfresponse_v1-00.pt`) which contains all the necessary information, such as the neural network weights and hyperparameters, the version number to choose the correct model, or the coefficients for the effective area or background rate model.
# 
# Another important task is the creation of compute pools. All PyTorch-related inference tasks are managed by a separate process for each chosen device. These processes need to be started or stopped, as shown in the example below.
# 
# All queries, such as evaluating the effective area or density as well as sampling, are passed through an `Approximation` object (which chooses the correct model and prepares the input) to a `Model` object (which handles the actual inference on the PyTorch device). For the response $R = A_\mathrm{eff}(\nu\lambda, E_i) \cdot P(E_m, \phi, \psi\chi|\nu\lambda, E_i)$, this includes:
# 
# 1. The effective area $A_\mathrm{eff}$, modeled with a spherical harmonics expansion. The latter are evaluated using the PyTorch backend of the `sphericart` library.
# 2. The probability density $P(E_m, \phi, \psi\chi|\nu\lambda, E_i)$, modeled with conditional-autoregressive-rational quadratic spline flows. The library used is called `normflows`.
# 
# The background model $B = R(t) \cdot P(E_m, \phi, \psi\chi|t)$ follows a very similar structure and therefore uses most of the same code. While the rate $R(t)$ is always computed on the CPU, the density $P$ is also modeled using `normflows`.
# 
# #### Folding
# 
# For the unbinned analysis, we need to calculate the total number of events we expect, $N$, and the expectation density, $\mathrm{d}N/\mathrm{d}t\mathrm{d}E_m\mathrm{d}\phi\mathrm{d}\psi\chi$. The latter is especially difficult to compute, since it requires folding the response with the flux model and therefore computing a numerical integral with enough precision for every event in the analysis. The background model $B$, on the other hand, is already the expectation density and requires no integration.
# 
# The folding can be performed using `UnbinnedThreeMLPointSourceResponseTrapz`. However, since $R$ has a low inference rate, `UnbinnedThreeMLPointSourceResponseIRFAdaptive` provides an optimized implementation. It uses the fact that many flux models are "well-behaved" (low order in a Taylor series). This means one can significantly reduce the integration error by concentrating most Gauss-Legendre integration nodes at peaks of the response and distributing the remaining ones in between. 
# 
# The exact parameters can be tuned by the user, but it probably won't be able to account for very complex spectral shapes. For each integration node, the response is evaluated once and cached, which takes most of the time. This way, during optimization, only the flux model changes and the fitting is fast.
# 
# Using `CachedUnbinnedThreeMLModelFolding` instead of `UnbinnedThreeMLModelFolding` adds the capability to save this cache and other parameters of `UnbinnedThreeMLPointSourceResponseIRFAdaptive`. These files can be shared with other scientists and loaded during another session to skip the expensive cache initialization.
# 
# ### Hardware Requirements
# 
# It is highly recommended to use a GPU, preferably by NVIDIA, for the inference. A CPU, even something like an AMD Threadripper, will only allow you to analyze a few orbits in a reasonable time. Also, note that the unbinned analysis requires you to have enough RAM, which increases as the number of events you include in the analysis increases. The batch size can be decreased to limit the maximum consumption.
# 
# ### Troubleshooting
# 
# #### Too many open files error
# 
# You might get this error while trying to run the tutorial: `RuntimeError: unable to open shared memory object </torch_...> in read-write mode: Too many open files (...)`
# 
# The quickest fix is to tell your OS to allow more open files. On Linux or macOS, you can check your current limit by running `ulimit -n` in your terminal. To increase it for the current session: `ulimit -n 65535`

# ## Example

# ### Basic Setup

# In[ ]:


import time
from pathlib import Path
from cosipy.util import fetch_wasabi_file
from cosipy.spacecraftfile import SpacecraftHistory
from astropy.time import Time

import astropy.units as u
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from histpy import Histogram

from threeML import Band, PointSource, Model, JointLikelihood, DataList

from cosipy.threeml.unbinned_model_folding import CachedUnbinnedThreeMLModelFolding
from cosipy.statistics import UnbinnedLikelihood
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.interfaces.expectation_interface import SumExpectationDensity

from cosipy.event_selection.time_selection import TimeSelector
from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSDistanceEventDataInSCFrameFromDC3Fits

from cosipy.response.ml.NFResponse import NFResponse
from cosipy.response.ml.nf_instrument_response_function import UnpolarizedNFFarFieldInstrumentResponseFunction
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized


from cosipy.threeml.ml.optimized_unbinned_folding import UnbinnedThreeMLPointSourceResponseIRFAdaptive

if __name__ == "__main__":

    #irf_mode = "nn"
    irf_mode = "hist"
    #irf_mode = "mixed"

    start_time = time.time()

    # In[ ]:


    data_path = Path("./") # Current path by default

    grb_data_path = data_path / "GRB_bn090424592_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
    fetch_wasabi_file('COSI-SMEX/DC3/Data/Sources/GRB_bn090424592_3months_unbinned_data_filtered_with_SAAcut.fits.gz', 
                      checksum = '08c1f1202d78df40a4aed3b785d273e0', output=str(grb_data_path))

    sc_orientation_path = data_path / "DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits"
    fetch_wasabi_file('COSI-SMEX/DC4/Data/Orientation/DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits', 
                      checksum = 'ca94ff1d7a73c1f41479aaf598807673', output=str(sc_orientation_path))

    rsp_path = data_path / "unpolarized_nfresponse_v1-01.pt"
    fetch_wasabi_file('COSI-SMEX/DC4/Data/Responses/unpolarized_nfresponse_v1-01.pt',
                      checksum = 'bf2d0c16eac5954fb56489480c2602ca', output=str(rsp_path))


    # Define the observation duration here. This GRB lasts approximately 10 seconds.

    # In[ ]:


    tstart = Time("2028-03-24 10:36:42.000")
    tstop = Time("2028-03-24 10:36:51.921")
    sc_orientation = SpacecraftHistory.open(sc_orientation_path)
    sc_orientation = sc_orientation.select_interval(tstart, tstop)


    # Typically, you would need to perform a time selection. Although this tutorial includes pre-selected files covering the full duration of the GRB, reducing the observation window is recommended if you are working with limited computational resources.

    # In[ ]:


    data_file = [grb_data_path]
    time_selector = TimeSelector(tstart = sc_orientation.tstart, tstop = sc_orientation.tstop)


    if irf_mode == 'hist':
        from cosipy.event_selection import DistanceSelector, ChainEventSelectors
        distance_selector = DistanceSelector(min_distance=1 * u.cm)
        selector = ChainEventSelectors(distance_selector, time_selector)

    elif irf_mode in ['nn', 'mixed']:

        # Mixed get the Aeff from nn, so no

        selector = time_selector

    else:
        raise RuntimeError(f"irf_mode {irf_mode} is not supported.")

    data = TimeTagEmCDSDistanceEventDataInSCFrameFromDC3Fits(data_file, selection=selector)


    # In[ ]:


    print(f"This analysis uses {data.nevents} Events")


    # `NFResponse` and `NFBackground`
    # - `devices`: Optional default devices. 
    #     - This example notebook defaults to CPU with the `["cpu"]` argument.
    #     - For GPU usage, provide a list of devices following standard PyTorch naming conventions. For example, a system with four CUDA-capable GPUs can be accessed using `["cuda:0", "cuda:1", "cuda:2", "cuda:3"]`.
    #         - You must install PyTorch with the CUDA version that matches your system. You can check your system's supported CUDA version by running the `nvidia-smi` command in your terminal.
    #         - Once you know your system version, verify which CUDA version PyTorch is currently using by running: `import torch; print(torch.version.cuda); print(torch.__version__)`.
    #         - If the versions do not match, reinstall PyTorch using the specific index URL: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu{iii}`, where `{iii}` is the CUDA version (e.g., cu121 for CUDA 12.1). Note that not all versions are available and you may need to select the closest compatible version.
    #     - Using the devices argument causes the compute pool to initialize and close automatically for each inference. This may produce unnecessary overhead unless used with functions like `UnbinnedThreeMLPointSourceResponseIRFAdaptive`, which handles pool management internally.
    #     - Alternatively, you can manually manage compute pools using `init_compute_pool`, `clean_compute_pool`, and `shutdown_compute_pool`.
    # - `compile_mode`: Optional from a list of options.
    #     - To ensure maximum compatibility, this tutorial defaults to `None`. 
    #     - For potential speed-ups, you can experiment with `density_compile_mode="default"` and `area_compile_mode="max-autotune-no-cudagraphs"` or other [PyTorch compilation modes](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

    # In[ ]:


    rsp = NFResponse(
        path_to_model=rsp_path,
        area_batch_size=300_000,
        density_batch_size=100_000,
        devices=["cpu"],
        area_compile_mode=None,
        density_compile_mode=None,
        show_progress=True)

    if irf_mode == 'hist':

        # Just new hist
        irf = IRFRelativeHistUnpolarized.from_h5(
            "/Users/imartin5/cosi/scratch/response_relative_coordinates/v4/ResponseContinuum.area.relative.nonsparse.h5")

    elif irf_mode == 'mixed':

        # Mix aeff through hist
        irf = IRFRelativeHistUnpolarized(
              Histogram.open("/Users/imartin5/cosi/scratch/response_relative_coordinates/v3/ResponseContinuum.area.relative.nonsparse.h5", "IRF"),
              aeff = Histogram.open("/Users/imartin5/cosi/scratch/response_relative_coordinates/v3/aeff_nside64_ei64_NNresponse.h5", 'aeff'))

    elif irf_mode == 'nn':

        irf = UnpolarizedNFFarFieldInstrumentResponseFunction(rsp)

    else:

        raise RuntimeError(f"irf_mode {irf_mode} is not supported.")


    # `UnbinnedThreeMLPointSourceResponseIRFAdaptive`
    # - `force_energy_node_caching`
    #     - Saves the energy nodes used for integration even when the batch size is small. This increases memory consumption but significantly speeds up expectation calculations during optimization.
    # - `reduce_memory`
    #     - Saves caches as `float32`, which can reduce the memory footprint. However, this will slow down expectation calculations. Note that for large batch sizes, this may actually increase memory consumption, so keep an eye out for related warnings.
    # - Other arguments
    #     - The class provides several getters and setters for parameters regarding integration nodes and batch sizes.

    # In[ ]:


    psr = UnbinnedThreeMLPointSourceResponseIRFAdaptive(
        data=data, 
        irf=irf, 
        sc_history=sc_orientation, 
        show_progress=True, 
        force_energy_node_caching=True, 
        reduce_memory=True) 

    psr.cache_batch_size = 100_000
    psr.integration_batch_size = 100_000


    # The default `Band`, `Powerlaw` and `PointSource` models can be slow during optimization (see the [discussion here](https://github.com/cositools/cosipy/discussions/492)). For larger datasets, consider implementing Torch-based versions to improve performance.

    # In[ ]:


    l = 21.293418951586656
    b = -42.384846594045484

    alpha = -1.021866
    beta = -2.762827
    xp = 159.9995 * u.keV
    piv = 100. * u.keV
    K = 5.2385389556 / u.cm / u.cm / u.s / u.keV

    spectrum = Band(beta=beta, K=K.value, piv=piv.value)

    spectrum.alpha.delta = 0.01
    spectrum.beta.delta = 0.01

    spectrum.alpha.value = alpha
    spectrum.xp.value = xp.value

    spectrum.xp.unit = xp.unit
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit


    # In[ ]:


    spectrum_inj = deepcopy(spectrum)


    # In[ ]:


    source = PointSource("GRB",
                         l=l,
                         b=b,
                         spectral_shape=spectrum)
    model = Model(source)


    # In[ ]:


    response = CachedUnbinnedThreeMLModelFolding(psr)


    # In[ ]:


    expectation_density = SumExpectationDensity(response)


    # In[ ]:


    like_fun = UnbinnedLikelihood(expectation_density)
    cosi = ThreeMLPluginInterface('cosi', like_fun, response)


    # In[ ]:


    plugins = DataList(cosi)
    like = JointLikelihood(model, plugins, verbose=True) # You can disable debugging


    # ### Initializing the Cache

    # Here you could load the cache

    # In[ ]:


    # response.load_caches(data_path / "GRB_tutorial")


    # The cache is initialized, which takes some time

    # In[ ]:


    print(f"Data Events: {data.nevents}\nExpected Events: {expectation_density.expected_counts():.2f}\nRelative Deviation {100 * (expectation_density.expected_counts()/data.nevents - 1):.3f} %")


    # Now you could save the cache.

    # In[ ]:


    response.save_caches(data_path / "GRB_tutorial")


    # ### Fitting

    # In[ ]:


    like.fit()


    # Now we can plot the result and compare it with the injected spectrum.

    # In[ ]:


    results = like.results

    parameters = {par.name: results.get_variates(par.path)
                  for par in results.optimized_model["GRB"].parameters.values()
                  if par.free}

    results_err = results.propagate(results.optimized_model["GRB"].spectrum.main.shape.evaluate_at, **parameters)


    # In[ ]:


    energy = np.geomspace(100*u.keV, 10*u.MeV).to_value(u.keV)

    flux_lo = np.zeros_like(energy)
    flux_median = np.zeros_like(energy)
    flux_hi = np.zeros_like(energy)
    flux_inj = np.zeros_like(energy)

    for i, e in enumerate(energy):
        flux = results_err(e)
        flux_median[i] = flux.median
        flux_lo[i], flux_hi[i] = flux.equal_tail_interval(cl=0.68)
        flux_inj[i] = spectrum_inj.evaluate_at(e)


    # In[ ]:


    #[magic commented out by run_tutorials.py] %matplotlib inline


    # In[ ]:


    fig, ax = plt.subplots(figsize = (9, 6))

    ax.plot(energy, energy**2 * flux_median, label = "Best fit")
    ax.fill_between(energy, energy**2 * flux_lo, energy*energy*flux_hi, alpha = .5, label = "Best fit (errors)")
    ax.plot(energy, energy**2 * flux_inj, color = 'black', ls = ":", label = "Injected")

    ax.semilogx()
    ax.semilogy()

    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel(r"$E^2 \frac{\mathrm{d}N}{\mathrm{d}E}$ [keV cm$^{-2}$ s$^{-1}$]")

    ax.legend();


    # In[ ]:

    print(f"Wall time: {time.time() - start_time:.2f} s")

    plt.show()





