Tutorials
=========

Tutorial for various components of the `cosipy` library. These are Python
notebooks that you can execute interactively.

List of tutorials and contents (WiP):

1. `Data format and handling <https://github.com/cositools/cosipy/tree/main/docs/tutorials/DataIO/DataIO_example.ipynb>`_
  
  - Explain the data format, binned and unbinned
  - Show how to bin it in both local and galactic coordinates
  - Show how to combine files.
  - Show how to inspect and plot it
    
2. `Spacecraft orientation and location <https://github.com/cositools/cosipy/tree/main/docs/tutorials/Point_source_resonse.ipynb>`_
  
  - Describe the contents for the raw SC file
  - Describe how to manipulate it —e.g. get a time range, rebin it.  
  - Explain the meaning of the dwell time map and how to obtain it
  - Explain the meaning of the scatt map and how to obtain it

3. `Detector response and signal expectation <https://github.com/cositools/cosipy/tree/main/docs/tutorials/DetectorResponse.ipynb>`_
  
  - Explain the format and meaning of the detector response
  - Show how to visualize it
  - Explain how to convolve the detector response with a point source model (location + spectrum) + spacecraft file to obtain the expected signal counts. Both in SC and galactic coordinates.
    
4. `TS Map: localizing a GRB <https://github.com/cositools/cosipy/tree/main/docs/tutorials/Parallel_TS_map_computation.ipynb>`_
  - Explain the TS calculation
  - Explain the meaning of the TS map and how to compute confidence contours
  - Compute a TS map, get the best location and estimate the error
    
5. `Fitting the spectrum of a GRB <https://github.com/cositools/cosipy/tree/main/docs/tutorials/spectral_fits/continuum_fit/grb/SpectralFit.ipynb>`_
  
  - Introduce 3ML and astromodels
  - Explain the likelihood. Reference previous example for data IO, SC file and response. 
  - Explain how the background is computed/fitted.
  - Fit a simple power law, assuming you know the time of the GRB
  - Show how to plot the result
  - Show how to compare the result with the data
    
6. `Fitting the spectrum of the Crab <https://github.com/cositools/cosipy/tree/main/docs/tutorials/spectral_fits/continuum_fit/crab/SpectralFit_Crab.ipynb>`_
  
  - Explain why we can’t work directly in SC coordinates, as for the GRB. 
  - Perform the fit. Here we need less explanation, since most of the machinery was already introduced.
  - DC Extended source model fitting
  - Explain how the extended source response is a convolution of multiple point sources, and the meaning of the sky model map
  - Describe how to pre-compute a response in galactic coordinates for all-sky. Explain also how to use it.
  - Fit the normalization of a simple model, assuming you know the shape and spectrum
  - Nice to have: free from spectral or shape parameters
    
7. `Image deconvolution <https://github.com/cositools/cosipy/tree/main/docs/tutorials/image_deconvolution/511keV/ScAttBinning/511keV-DC2-ScAtt-ImageDeconvolution.ipynb>`_
  - Explain the RL algorithm. Reference the previous example. Explain difference with TS map.
  - Explain the scatt binning and its advantages/disadvantages
  - Fit the Crab
  - Fit the 511 diffuse emission or similar.
    
8. TODO: Source injector
  - Nice to have: allow theorist to test the sensitivity of their models

.. toctree::
   :maxdepth: 1

   Data format and handling <DataIO/DataIO_example.ipynb>
   Spacecraft orientation and location <Point_source_resonse.ipynb>
   Detector response and signal expectation <DetectorResponse.ipynb>
   TS Map: localizing a GRB <Parallel_TS_map_computation.ipynb>
   Fitting the spectrum of a GRB <spectral_fits/continuum_fit/grb/SpectralFit.ipynb>
   Fitting the spectrum of the Crab <spectral_fits/continuum_fit/crab/SpectralFit_Crab.ipynb>
   Image deconvolution <image_deconvolution/511keV/ScAttBinning/511keV-DC2-ScAtt-ImageDeconvolution.ipynb>
