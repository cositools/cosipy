from cosipy.threeml.custom_functions import SpatialTemplate_2D_Healpix
from mhealpy import HealpixMap
import astropy.units as u
import numpy as np

def test_2DTemplateHealpix(tmp_path):
  
  # Test template function with healpix map
  skymap = HealpixMap(nside=8, scheme="ring", dtype=float, coordsys='G')
  skymap[:] = 1

  # normalise to the pixel area
  area = skymap.pixarea().value
  skymap[:] = skymap[:]/(np.sum(skymap)*area)

  # write the fits file
  skymap.write_map(tmp_path/"HealpixMap_test.fits", overwrite=True)
  
  #test the init of the function
  spatial_shape =  SpatialTemplate_2D_Healpix(fits_file=tmp_path/"HealpixMap_test.fits")

  #test evaluate
  value = spatial_shape.evaluate(0*u.deg, 0*u.deg, spatial_shape.K.value, spatial_shape.hash.value)

  #test boundaries
  boundaries = spatial_shape.get_boundaries()

  #test spatial integral
  integral = spatial_shape.get_total_spatial_integral()
