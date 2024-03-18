Tutorials
=========

This is a series of tutorials explaining step by step the various components of the `cosipy` library and how to use it. Although they are rendered as a webpage here, these are interactive Python notebooks (ipynb) that you can execute and modify, distributed as part of the cosipy repository.

If you are interested instead of the description of each class and method, please see our `API <../api/index.html>`_ section.

List of tutorials and contents, as a link to the corresponding Python notebook in the repository:

1. Data format and handling `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/DataIO/DataIO_example.ipynb>`_
  
  - Data format, binned and unbinned
  - Binning the data in both local and galactic coordinates
  - Combining files.
  - Inspecting and plotting the data
    
2. Spacecraft orientation and location `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/response/SpacecraftFile.ipynb>`_
  
  - SC file format and manipulation it —e.g. get a time range, rebin it.  
  - The dwell time map and how to obtain it
  - Generate point source response and export to the format that can be read by XSPEC
  - The scatt map and how to obtain it

3. Detector response and signal expectation `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/response/DetectorResponse.ipynb>`_
  
  - Explanation of the detector response format and meaning
  - Visualizing the response
  - Convolving the detector response with a point source model (location + spectrum) + spacecraft file to obtain the expected signal counts. Both in SC and galactic coordinates.
    
4. TS Map: localizing a GRB `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/ts_map/Parallel_TS_map_computation_DC2.ipynb>`_
  - TS calculation
  - Meaning of the TS map and how to compute confidence contours
  - Computing a TS map, getting the best location and estimating the error
    
5. Fitting the spectrum of a GRB `(ipynb) <https://github.com/cositools/cosipy/blob/main/docs/tutorials/spectral_fits/continuum_fit/grb/SpectralFit_GRB.ipynb>`_
  
  - Introduction to 3ML and astromodels
  - Likelihood analysis. 
  - Mechanics of background estimation.
  - Fitting a simple power law, assuming you know the time of the GRB
  - Plotting the result
  - Comparing the result with the data
    
6. Fitting the spectrum of the Crab `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/spectral_fits/continuum_fit/crab/SpectralFit_Crab.ipynb>`_
  
  - Analysing a continuous source transiting in the sky.

7. Extended source model fitting `(ipynb) <https://github.com/cositools/cosipy/blob/main/docs/tutorials/spectral_fits/extended_source_fit/diffuse_511_spectral_fit.ipynb>`_
   
  - Obtaining the extended source response as a convolution of multiple point sources
  - Pre-computing a response in galactic coordinates for all-sky
  - Fitting an extended source
    
8. Image deconvolution `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/image_deconvolution/511keV/ScAttBinning/511keV-DC2-ScAtt-ImageDeconvolution.ipynb>`_
  - Explain the RL algorithm. Reference the previous example. Explain the difference with a TS map.
  - Scatt binning and its advantages/disadvantages
  - Fitting the 511 diffuse emission.
    
9. TODO: Source injector `(ipynb) <https://github.com/cositools/cosipy/tree/main/docs/tutorials/source_injector/GRB_source_injector.ipynb`
  - Nice to have: allow theorist to test the sensitivity of their models

.. warning::
   Under construction. Some of the explanations described above might be missing. However, the notebooks are fully functional. If you have a question not yet covered by the tutorials, please discuss `issue <https://github.com/cositools/cosipy/discussions>`_ so we can prioritize it.
    
.. toctree::
   :maxdepth: 1

   Data format and handling <DataIO/DataIO_example.ipynb>
   response/SpacecraftFile.ipynb
   Detector response and signal expectation <response/DetectorResponse.ipynb>
   TS Map: localizing a GRB <ts_map/Parallel_TS_map_computation_DC2.ipynb>
   Fitting the spectrum of a GRB <spectral_fits/continuum_fit/grb/SpectralFit_GRB.ipynb>
   Fitting the spectrum of the Crab <spectral_fits/continuum_fit/crab/SpectralFit_Crab.ipynb>
   Extended source model fitting <spectral_fits/extended_source_fit/diffuse_511_spectral_fit.ipynb>
   Image deconvolution <image_deconvolution/511keV/ScAttBinning/511keV-DC2-ScAtt-ImageDeconvolution.ipynb>


