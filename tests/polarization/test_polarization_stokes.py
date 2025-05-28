import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from scoords import SpacecraftFrame

from cosipy.polarization.polarization_stokes import PolarizationStokes
from cosipy.spacecraftfile import SpacecraftFile
from cosipy import UnBinnedData
from cosipy.threeml.custom_functions import Band_Eflux
from cosipy import test_data

analysis = UnBinnedData(test_data.path / 'polarization_data.yaml')
data = analysis.get_dict_from_hdf5(test_data.path / 'polarization_data.hdf5')
response_path = test_data.path / 'test_polarization_response_dense.h5'
sc_orientation = SpacecraftFile.parse_from_file(test_data.path / 'polarization_ori.ori')
attitude = sc_orientation.get_attitude()[0]

a = 10. * u.keV
b = 10000. * u.keV
alpha = -1.
beta = -2.
ebreak = 350. * u.keV 
K = 50. / u.cm / u.cm / u.s
spectrum = Band_Eflux(a = a.value,
                      b = b.value,
                      alpha = alpha,
                      beta = beta,
                      E0 = ebreak.value,
                      K = K.value)
spectrum.a.unit = a.unit
spectrum.b.unit = b.unit
spectrum.E0.unit = ebreak.unit
spectrum.K.unit = K.unit

source_direction = SkyCoord(0, 70, representation_type='spherical', frame=SpacecraftFrame(attitude=attitude), unit=u.deg)

def test_stokes_polarization():

    source_photons = PolarizationStokes(source_direction, spectrum, data, 
                                        response_path, sc_orientation, background=None,
                                        response_convention='RelativeZ')

    qs, us = source_photons.compute_data_pseudo_stokes(show=False)

    average_mu = source_photons._mu100['mu']

    mdp99 = source_photons._mdp99

    polarization = source_photons.calculate_polarization(qs, us, average_mu['mu'], 
                                                         bkg_qs=None, bkg_us=None, show_plots=True, 
                                                         mdp=mdp99)
        
    assert np.allclose([polarization['fraction']*100, polarization['fraction uncertainty']*100,
                        polarization['angle'].angle.degree, polarization['angle uncertainty'].degree],
                        [13.73038868282377, 2.1295224814008353, np.degrees(1.4851296518928818),np.degrees(0.07562763316088744)], atol=[1.0, 0.5, 1.0, 0.1])


