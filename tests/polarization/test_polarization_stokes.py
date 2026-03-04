import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from scoords import SpacecraftFrame

from cosipy.polarization.polarization_stokes import PolarizationStokes, rotate_points_to_x_axis
from cosipy.spacecraftfile import SpacecraftFile
from cosipy import UnBinnedData
from cosipy.threeml.custom_functions import Band_Eflux
from cosipy import test_data

analysis = UnBinnedData(test_data.path / 'polarization_data.yaml')
data = analysis.get_dict_from_hdf5(test_data.path / 'polarization_data.hdf5')
response_path = test_data.path / 'test_polarization_response.h5'
sc_orientation = SpacecraftFile.open(test_data.path / 'polarization_ori.fits')
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

    bin_edges = Angle(np.linspace(-np.pi, np.pi, 10), unit=u.rad)
    source_photons = PolarizationStokes(source_direction, spectrum, data, 
                                        response_path, sc_orientation, background=None, 
                                        response_convention='RelativeZ', asad_bin_edges=bin_edges, show_plots=False)
    
    average_mu = source_photons._mu100['mu']
    mdp99 = source_photons._mdp99
    bkg_duration = source_photons.get_background_duration()
    print('Bkg duration (should be 0):', bkg_duration)

    qs, us = source_photons.compute_data_pseudo_stokes(show_plots=False)
    polarization = source_photons.calculate_polarization(qs, us, average_mu, mdp=mdp99,
                                                         bkg_qs=None, bkg_us=None, show_plots=False)
    Pol_frac = polarization['fraction'] * 100
    Pol_angl = polarization['angle'].angle.degree

    test_pd, test_pa = 0.8, 90
    test_q, test_u = rotate_points_to_x_axis(test_pd, np.radians(test_pa))
    print('Testing rotate_points_to_x_axis (returns Q,U given PD,PA)', test_q, test_u)

    assert np.allclose([average_mu, mdp99, Pol_frac, Pol_angl], [0.19, 0.22, 185, 82], atol=[0.1, 0.1, 5, 10])