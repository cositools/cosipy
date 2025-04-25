from cosipy.threeml.custom_functions import GalpropHealpixModel
from cosipy import test_data
import numpy as np
from  astropy.coordinates import Galactic, ICRS
from astromodels import ExtendedSource, Model

def test_GalpropHealpixModel(tmp_path):

    input_fits = test_data.path/'ics_isotropic_healpix_54_0780000f.gz'
    galprop_model = GalpropHealpixModel()
    galprop_model.set_version(54)
    galprop_model.load_file(input_fits)

    assert galprop_model._fitsfile == input_fits
    assert galprop_model._gal_version == 54
    assert galprop_model._frame == "galactic"
    assert galprop_model.K.value == 1
    assert str(galprop_model.K.unit) == ''

    # Define some Galactic coordinates (l, b) and energy values
    l = np.array([0, 10, 30, 60])       # Galactic longitude in degrees
    b = np.array([0, 5, -5, 10])        # Galactic latitude in degrees
    e = np.array([0.2, 1, 10])          # Energies in MeV

    # Evaluate the model
    flux = galprop_model.evaluate(l, b, e, 1)

    assert str(flux.unit) == '1 / (MeV s sr cm2)'
    assert flux.shape == (4,3) # (4 positions, 3 energies)

    # Change frame:
    icrs_frame = ICRS()
    galprop_model.set_frame(icrs_frame)
    assert galprop_model._frame == 'icrs'
    
    # Re-evaluate b/c coords will now be converted to galactic,
    # since coords are expected to match frame:
    flux_icrs = galprop_model.evaluate(l, b, e, 1)
    assert np.array_equal(flux,flux_icrs) == False 

    # Test integration method:
    galprop_model.get_total_spatial_integral(e, avg_int=True, nside=2)

    # Write output:
    src = ExtendedSource("galprop_source", spatial_shape=galprop_model)
    model = Model(src)
    model.save(tmp_path/"galprop_model.yaml", overwrite=True)
