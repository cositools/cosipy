import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from scoords import SpacecraftFrame

from cosipy.polarization_fitting import PolarizationASAD
from cosipy.polarization.conventions import IAUPolarizationConvention, MEGAlibRelativeZ
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy import BinnedData
from cosipy.threeml.custom_functions import Band_Eflux
from cosipy import test_data

analysis = BinnedData(test_data.path / 'polarization_data.yaml')
unbinned_data = analysis.get_dict_from_hdf5(test_data.path / 'polarization_data.hdf5')
analysis.get_binned_data(unbinned_data = test_data.path / 'polarization_data.hdf5')
binned_data = analysis.binned_data

response_path = test_data.path / 'test_polarization_response.h5'
sc_orientation = SpacecraftHistory.open(test_data.path / 'polarization_ori.fits')
attitude = sc_orientation.attitude[0]

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

source_direction = SkyCoord(0, 70, representation_type='spherical', unit=u.deg,
                            frame=SpacecraftFrame(attitude=attitude))

bin_edges = Angle(np.linspace(-np.pi, np.pi, 10), unit=u.rad)

background = {
    'Psi local': np.array([0, 0]),
    'Chi local': np.array([0, 0]),
    'Psi galactic': np.array([0, 0]),
    'Chi galactic': np.array([0, 0]),
    'Energies': np.array([300., 300.]),
    'TimeTags': np.array([1., 2.])
}

def test_spacecraft_fit():

    # ASAD from unbinned data
    polarization_spacecraft = PolarizationASAD(source_direction,
                                               spectrum, bin_edges,
                                               unbinned_data, background,
                                               sc_orientation, response_path,
                                               response_convention='RelativeZ',
                                               fit_convention=MEGAlibRelativeZ(attitude=attitude))

    polarization_fit_spacecraft = polarization_spacecraft.fit()

    assert np.allclose([polarization_fit_spacecraft['fraction'],
                        polarization_fit_spacecraft['fraction uncertainty'],
                        polarization_fit_spacecraft['angle'].angle.rad,
                        polarization_fit_spacecraft['angle uncertainty'].rad],
                       [0.8248301194006322, 0.7889544546660269,
                        1.5645814961205469, 0.5143016570410734],
                       atol=[0.2, 0.1, 0.2, 0.1])

    # ASAD from binned data
    polarization_spacecraft = PolarizationASAD(source_direction,
                                               spectrum, bin_edges,
                                               binned_data, background,
                                               sc_orientation, response_path,
                                               response_convention='RelativeZ',
                                               fit_convention=MEGAlibRelativeZ(attitude=attitude))

    polarization_fit_spacecraft = polarization_spacecraft.fit()

    assert np.allclose([polarization_fit_spacecraft['fraction'],
                        polarization_fit_spacecraft['fraction uncertainty'],
                        polarization_fit_spacecraft['angle'].angle.rad,
                        polarization_fit_spacecraft['angle uncertainty'].rad],
                       [0.9452187271167997, 0.9328483275998886,
                        1.993361180746714, 0.6416512077658346],
                       atol=[0.2, 0.1, 0.2, 0.1])

def test_icrs_fit():

    # ASAD from unbinned data
    polarization_icrs = PolarizationASAD(source_direction.transform_to('galactic'),
                                         spectrum, bin_edges,
                                         unbinned_data, background,
                                         sc_orientation, response_path,
                                         response_convention='RelativeZ')

    polarization_fit_icrs = polarization_icrs.fit()

    assert np.allclose([polarization_fit_icrs['fraction'],
                        polarization_fit_icrs['fraction uncertainty'],
                        polarization_fit_icrs['angle'].angle.rad,
                        polarization_fit_icrs['angle uncertainty'].rad],
                       [1.496745801986812, 0.8751758133432178,
                        1.84399578798795, 0.3812557749920544],
                       atol=[0.2, 0.1, 0.2, 0.1])

    # ASAD from binned data
    polarization_icrs = PolarizationASAD(source_direction.transform_to('galactic'),
                                         spectrum, bin_edges,
                                         binned_data, background,
                                         sc_orientation, response_path,
                                         response_convention='RelativeZ')

    polarization_fit_icrs = polarization_icrs.fit()

    assert np.allclose([polarization_fit_icrs['fraction'],
                        polarization_fit_icrs['fraction uncertainty'],
                        polarization_fit_icrs['angle'].angle.rad,
                        polarization_fit_icrs['angle uncertainty'].rad],
                       [2.02118419504387, 0.7661035298627569,
                        1.6238519333293382, 0.22647693546905168],
                       atol=[0.2, 0.1, 0.2, 0.1])
