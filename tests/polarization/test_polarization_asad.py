import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from scoords import SpacecraftFrame

from cosipy.polarization import PolarizationASAD, calculate_uncertainties
from cosipy.polarization.conventions import IAUPolarizationConvention
from cosipy.spacecraftfile import SpacecraftFile
from cosipy import UnBinnedData
from cosipy.threeml.custom_functions import Band_Eflux
from cosipy import test_data

analysis = UnBinnedData(test_data.path / 'polarization_data.yaml')
data = analysis.get_dict_from_hdf5(test_data.path / 'polarization_data.hdf5')
response_path = test_data.path / 'test_polarization_response.h5'
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
    
polarization = PolarizationASAD(source_direction, spectrum, response_path, sc_orientation)

bin_edges = Angle(np.linspace(-np.pi, np.pi, 10), unit=u.rad)

def test_calculate_uncertainties():

    assert np.allclose(calculate_uncertainties([1, 2, 3, 5, 10, 50, 100, 1000, 1e4, 1e5]), 
                        np.array([[0.82724622, 1.29181456, 1.63270469, 2.15969114, 3.16227766, 7.07106781, 10., 31.6227766, 100., 316.22776602], 
                                  [2.29952656, 2.63785962, 2.91818583, 3.38247265, 3.16227766, 7.07106781, 10., 31.6227766, 100., 316.22776602]]))
    
def test_convolve_spectrum():

    assert 'Pol' in polarization._expectation.axes.labels

    assert polarization._expectation.unit.is_equivalent('keV')

    assert np.allclose(polarization._expectation.project('Em').contents[0], 7657.331992884072, atol=5.0)

    assert np.allclose([polarization._azimuthal_angle_bins[5].rad, polarization._azimuthal_angle_bins[12].rad, 
                        polarization._azimuthal_angle_bins[25].rad, polarization._azimuthal_angle_bins[40].rad], 
                       [-3.083541411881164, -1.5707963267948966, 0.371190531909648, 0.9731904991340073])

def test_calculate_azimuthal_scattering_angle():

    assert np.allclose([polarization.calculate_azimuthal_scattering_angle(0, 0).rad, polarization.calculate_azimuthal_scattering_angle(np.pi/5, 0).rad,
                        polarization.calculate_azimuthal_scattering_angle(np.pi/2, -np.pi/7).rad, polarization.calculate_azimuthal_scattering_angle(np.pi/3, 3*np.pi/2).rad],
                        [1.5707963267948966, -1.5707963267948966, -1.097213841102146, 0.1949572818104049])
    
def test_polarization_asad_init():

    source_direction_galactic = SkyCoord(0, 0, representation_type='spherical', frame='galactic', attitude=attitude, unit=u.deg)
    polarization_iau = PolarizationASAD(source_direction_galactic, spectrum, response_path, sc_orientation, fit_convention=IAUPolarizationConvention())
    
def test_polarization_fit():

    azimuthal_angles = polarization.calculate_azimuthal_scattering_angles(data)

    assert np.allclose([azimuthal_angles[150].rad, azimuthal_angles[3275].rad, azimuthal_angles[5780].rad, azimuthal_angles[7050].rad], 
                       [-0.14553958772561387, 2.848889133943549, -2.2494887481804238, 1.2037585274409062])

    asad = polarization.create_asad(azimuthal_angles, bin_edges)

    assert np.allclose([asad['counts'][0], asad['counts'][3], asad['counts'][6], asad['counts'][8]], [1114, 727, 587, 883])
    
    assert np.allclose([asad['uncertainties'][0][0], asad['uncertainties'][1][3], 
                        asad['uncertainties'][0][6], asad['uncertainties'][1][8]], 
                       [33.37663853655727, 26.962937525425527, 24.228082879171435, 29.715315916207253])
    
    asad_int_bins = polarization.create_asad(azimuthal_angles, 10)

    assert np.allclose([asad_int_bins['counts'][0], asad_int_bins['counts'][3], asad_int_bins['counts'][6], asad_int_bins['counts'][8]], [1114, 727, 587, 883])
    
    assert np.allclose([asad_int_bins['uncertainties'][0][0], asad_int_bins['uncertainties'][1][3], 
                        asad_int_bins['uncertainties'][0][6], asad_int_bins['uncertainties'][1][8]], 
                       [33.37663853655727, 26.962937525425527, 24.228082879171435, 29.715315916207253])

    asad_unpolarized = polarization.create_unpolarized_asad()

    assert np.allclose([asad_unpolarized['counts'][0], asad_unpolarized['counts'][3], asad_unpolarized['counts'][6], asad_unpolarized['counts'][8]], 
                       [932.4137328357068, 906.8283258620892, 794.565967927462, 999.309350107182])
    
    assert np.allclose([asad_unpolarized['uncertainties'][0][0], asad_unpolarized['uncertainties'][1][3], 
                       asad_unpolarized['uncertainties'][0][6], asad_unpolarized['uncertainties'][1][8]], 
                       [30.53545042791586, 30.11359038477626, 28.188046543303813, 31.61185458189984])
    
    asad_unpolarized_int_bins = polarization.create_unpolarized_asad(bins=10)

    assert np.allclose([asad_unpolarized_int_bins['counts'][0], asad_unpolarized_int_bins['counts'][3], 
                        asad_unpolarized_int_bins['counts'][6], asad_unpolarized_int_bins['counts'][8]], 
                       [932.4137328357068, 906.8283258620892, 794.565967927462, 999.309350107182])
    
    assert np.allclose([asad_unpolarized_int_bins['uncertainties'][0][0], asad_unpolarized_int_bins['uncertainties'][1][3], 
                       asad_unpolarized_int_bins['uncertainties'][0][6], asad_unpolarized_int_bins['uncertainties'][1][8]], 
                       [30.53545042791586, 30.11359038477626, 28.188046543303813, 31.61185458189984])
    
    asad_unpolarized_bin_list = polarization.create_unpolarized_asad(bins=bin_edges)

    assert np.allclose([asad_unpolarized_bin_list['counts'][0], asad_unpolarized_bin_list['counts'][3], 
                        asad_unpolarized_bin_list['counts'][6], asad_unpolarized_bin_list['counts'][8]], 
                       [932.4137328357068, 906.8283258620892, 794.565967927462, 999.309350107182])
    
    assert np.allclose([asad_unpolarized_bin_list['uncertainties'][0][0], asad_unpolarized_bin_list['uncertainties'][1][3], 
                       asad_unpolarized_bin_list['uncertainties'][0][6], asad_unpolarized_bin_list['uncertainties'][1][8]], 
                       [30.53545042791586, 30.11359038477626, 28.188046543303813, 31.61185458189984])
    
    asad_polarized = polarization.create_polarized_asads()
    
    assert np.allclose([asad_polarized['counts'][0][0], asad_polarized['counts'][1][3], asad_polarized['counts'][2][6], asad_polarized['counts'][3][8]], 
                       [229.70809288956985, 231.350661957243, 197.93859480541153, 243.6223882118957])
    
    assert np.allclose([asad_polarized['uncertainties'][0][0][0], asad_polarized['uncertainties'][1][1][3], 
                        asad_polarized['uncertainties'][2][0][6], asad_polarized['uncertainties'][3][1][8]], 
                       [15.1561239401626, 15.210215710411308, 14.069065171695366, 15.608407612946802])
    
    asad_polarized_int_bins = polarization.create_polarized_asads(bins=10)
    
    assert np.allclose([asad_polarized_int_bins['counts'][0][0], asad_polarized_int_bins['counts'][1][3], 
                        asad_polarized_int_bins['counts'][2][6], asad_polarized_int_bins['counts'][3][8]], 
                       [229.70809288956985, 231.350661957243, 197.93859480541153, 243.6223882118957])
    
    assert np.allclose([asad_polarized_int_bins['uncertainties'][0][0][0], asad_polarized_int_bins['uncertainties'][1][1][3], 
                        asad_polarized_int_bins['uncertainties'][2][0][6], asad_polarized_int_bins['uncertainties'][3][1][8]], 
                       [15.1561239401626, 15.210215710411308, 14.069065171695366, 15.608407612946802])
    
    asad_polarized_bin_list = polarization.create_polarized_asads(bins=bin_edges)
    
    assert np.allclose([asad_polarized_bin_list['counts'][0][0], asad_polarized_bin_list['counts'][1][3], 
                        asad_polarized_bin_list['counts'][2][6], asad_polarized_bin_list['counts'][3][8]], 
                       [229.70809288956985, 231.350661957243, 197.93859480541153, 243.6223882118957])
    
    assert np.allclose([asad_polarized_bin_list['uncertainties'][0][0][0], asad_polarized_bin_list['uncertainties'][1][1][3], 
                        asad_polarized_bin_list['uncertainties'][2][0][6], asad_polarized_bin_list['uncertainties'][3][1][8]], 
                       [15.1561239401626, 15.210215710411308, 14.069065171695366, 15.608407612946802])

    mu_100 = polarization.calculate_mu100(asad_polarized, asad_unpolarized)

    assert np.allclose([mu_100['mu'], mu_100['uncertainty']], 
                       [0.02068036893603115, 9.3940548992881e-07], atol=[0.01, 1e-7])
    
    asad_corrected = polarization.correct_asad(asad, asad_unpolarized)

    assert np.allclose([asad_corrected['counts'][0], asad_corrected['counts'][3], asad_corrected['counts'][6], asad_corrected['counts'][8]], 
                       [1.230972371153726, 0.826002019945234, 0.7611669401887197, 0.9104005858437615])
    
    assert np.allclose([asad_corrected['uncertainties'][0][0], asad_corrected['uncertainties'][1][3], 
                        asad_corrected['uncertainties'][0][6], asad_corrected['uncertainties'][1][8]], 
                       [0.05463841596540586, 0.04112013917034527, 0.04142682980425702, 0.04204822825061025])
    
    polarization_fit = polarization.fit(mu_100, asad_corrected['counts'], bounds=([0, 0, 0], [np.inf,np.inf,np.pi]), sigma=asad_corrected['uncertainties'])

    assert np.allclose([polarization_fit['fraction'], polarization_fit['fraction uncertainty'], 
                        polarization_fit['angle'].angle.rad, polarization_fit['angle uncertainty'].rad], 
                       [15.270059610935844, 2.3931615504423474, 1.53994098472832, 0.07217196641713962], atol=[1.0, 0.5, 1.0, 0.1])