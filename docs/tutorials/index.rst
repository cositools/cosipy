Tutorials
=========

This is a series of tutorials explaining step by step the various components of the `cosipy` library and how to use it. Although they are rendered as a webpage here, these are interactive Python notebooks (ipynb) that you can execute and modify, distributed as part of the cosipy repository. You can download them using the links below, or by cloning the whole repository running :code:`git clone https://github.com/cositools/cosipy`.

If you are interested instead of the description of each class and method, please see our `API <../api/index.html>`_ section.

See also `COSI's second data challenge <https://github.com/cositools/cosi-data-challenge-2>`_ for the scientific description of the simulated data used in the tutorials, as well as an explanation of the statistical tools used by cosipy.

List of tutorials and contents, as a link to the corresponding Python notebook in the repository:

1. Data format and handling `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/DataIO/DataIO_example.ipynb>`_
  
  - Data format, binned and unbinned
  - Binning the data in both local and galactic coordinates
  - Combining files.
  - Inspecting and plotting the data
    
2. Spacecraft orientation and location `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/response/SpacecraftHistory.ipynb>`_
  
  - SC file format and manipulation it —e.g. get a time range, rebin it.  
  - The dwell time map and how to obtain it
  - Generate point source response and export to the format that can be read by XSPEC
  - The scatt map and how to obtain it

3. Detector response and signal expectation `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/response/DetectorResponse.ipynb>`_
  
  - Explanation of the detector response format and meaning
  - Visualizing the response
  - Convolving the detector response with a point source model (location + spectrum) + spacecraft file to obtain the expected signal counts. Both in SC and galactic coordinates.
    
4. TS Map: localizing a GRB `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/ts_map/TS_map_fitting.ipynb>`_
  - TS calculation
  - Meaning of the TS map and how to compute confidence contours
  - Computing a TS map, getting the best location and estimating the error
    
5. Fitting the spectrum of the Crab (binned) `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/spectral_fits/continuum_fit/crab/SpectralFit_Crab.ipynb>`_
  
  - Introduction to 3ML and astromodels
  - Likelihood analysis. 
  - Mechanics of background estimation.
  - Plotting the result
  - Comparing the result with the data
  - Analysing a continuous source transiting in the sky.

6. Fitting the spectrum of a GRB (unbinned) `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/spectral_fits/continuum_fit/grb/example_grb_fit_normalizing_flows.ipynb>`_

  - Introduction to unbinned analysis using normalizing flows.
  - Setting up the neural-network response and background approximations.
  - Initializing and saving adaptive integration caches for faster folding.
  - Performing the unbinned likelihood fit for a GRB point source.
  - Plotting the result and comparing it with the injected spectrum.

7. Extended source model fitting `(ipynb) <https://github.com/cositools/cosipy/blob/main/docs/tutorials/spectral_fits/extended_source_fit/diffuse_511_spectral_fit.ipynb>`_
   
  - Obtaining the extended source response as a convolution of multiple point sources
  - Pre-computing a response in galactic coordinates for all-sky
  - Fitting an extended source
    
8. Image deconvolution `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/image_deconvolution/511keV-Galactic-ImageDeconvolution.ipynb>`_
  - Explain the RL algorithm. Reference the previous example. Explain the difference with a TS map.
  - Fitting the 511 diffuse emission.
  - Analyze data in the Compton data space with galactic coordinates.
  - Link to a notebook using Scatt binning which shows its advantages/disadvantages.
    
9. Source injector `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/source_injector/Point_source_injector.ipynb>`_
  - Convolve the response, point source model and orientation to obtain the mock data.
  - More types of source (e,g. extended source and polarization) will be suppored.

10. Continuum background estimation `(ipynb) <https://github.com/cositools/cosipy/blob/develop/docs/tutorials/background_estimation/continuum_estimation/BG_estimationNN_example.ipynb>`_
  - Estimating the continuum background from the data. 

11. Line background estimation `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/background_estimation/line_background/line_background_estimation_example_notebook.ipynb>`_
  - Estimating the background from neighboring energy bins.

12. Polarization (ASAD method) `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/polarization/ASAD_method.ipynb>`_
  - Estimating the polarization degree and angle of a GRB using the Azimuthal Scattering Angle Distribution (ASAi)


13. Transient background estimation `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/background_estimation/transient_background/Transient_background_example.ipynb>`_
    
  - Estimating the background for transients.

14. Polarization (ASAD method) `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/polarization/ASAD_method.ipynb>`_
  - Estimating the polarization degree and angle of a GRB using the Azimuthal Scattering Angle Distribution (ASAD)

15. Polarization (Stokes parameters method) `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/polarization/Stokes_method.ipynb>`_
  - Estimating the polarization of a GRB using Stokes parameters

16. Phase-resolved analysis `(ipynb) <https://github.com/cositools/cosipy/blob/develop/docs/tutorials/phase_resolved_analysis/example_notebook.ipynb>`_
   - Folding event data and scaling mission exposure using pulsar timing models (ephemeris).

.. warning::
   Under construction. Some of the explanations described above might be missing. However, the notebooks are fully functional. If you have a question not yet covered by the tutorials, please discuss `issue <https://github.com/cositools/cosipy/discussions>`_ so we can prioritize it.
    
.. toctree::
   :maxdepth: 1

   Data format and handling <DataIO/DataIO_example.ipynb>
   response/SpacecraftHistory.ipynb
   Detector response and signal expectation <response/DetectorResponse.ipynb>
   TS Map: localizing a GRB <ts_map/Parallel_TS_map_computation.ipynb>
   Fitting the spectrum of the Crab (binned) <spectral_fits/continuum_fit/crab/SpectralFit_Crab.ipynb>
   Fitting the spectrum of a GRB (unbinned) <spectral_fits/continuum_fit/grb/example_grb_fit_normalizing_flows.ipynb>
   Extended source model fitting <spectral_fits/extended_source_fit/diffuse_511_spectral_fit.ipynb>
   Image deconvolution <image_deconvolution/511keV-Galactic-ImageDeconvolution.ipynb>
   Source injector <source_injector/Point_source_injector.ipynb>
   Continuum Background Estimation <background_estimation/continuum_estimation/BG_estimationNN_example.ipynb>
   Line background estimation <background_estimation/line_background/line_background_estimation_example_notebook.ipynb>
   Polarization (ASAD method) <polarization/ASAD_method.ipynb>
   Transient background estimation <background_estimation/transient_background/Transient_background_example.ipynb>
   Phase-resolved analysis <phase_resolved_analysis/example_notebook.ipynb>
