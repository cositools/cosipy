import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from scoords import SpacecraftFrame

from cosipy.polarization import PolarizationStokes
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

# bin_edges = Angle(np.linspace(-np.pi, np.pi, 10), unit=u.rad)

background = {'Psi local': [0, 0], 'Chi local': [0, 0], 'Psi galactic': [0, 0], 'Chi galactic': [0, 0], 'Energies': [300., 300.], 'TimeTags': [1., 2.]}

def test_stokes_polarization():

    source_photons = PolarizationStokes(source_direction, spectrum, data, background, 
                                        response_path, sc_orientation, 
                                        response_convention='RelativeX')

    qs, us = source_photons.compute_data_pseudo_stokes(show=True)
    bkg_qs, bkg_us = source_photons.compute_background_pseudo_stokes(show=True)

    average_mu = source_photons.calculate_average_mu100(show_plots=True) 

    polarization = source_photons.calculate_polarization(qs, us, bkg_qs, bkg_us, 
                                                         average_mu['mu'], show_plots=True, 
                                                         mdp=source_photons._mdp99)