import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from scoords import SpacecraftFrame

from cosipy.polarization_fitting import PolarizationStokes
from cosipy.polarization.conventions import IAUPolarizationConvention, MEGAlibRelativeZ
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy import UnBinnedData
from cosipy.threeml.custom_functions import Band_Eflux
from cosipy import test_data

analysis = UnBinnedData(test_data.path / 'polarization_data.yaml')
unbinned_data = analysis.get_dict_from_hdf5(test_data.path / 'polarization_data.hdf5')

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

def test_stokes_spacecraft_fit():

    polarization_spacecraft = PolarizationStokes(source_direction,
                                                 spectrum, bin_edges,
                                                 unbinned_data,
                                                 sc_orientation, response_path,
                                                 background=background,
                                                 fit_convention=MEGAlibRelativeZ(attitude=attitude),
                                                 show_plots=True)

    polarization_fit_spacecraft = polarization_spacecraft.fit(show_plots=True)

    print(polarization_fit_spacecraft['fraction'],
          polarization_fit_spacecraft['fraction uncertainty'],
          polarization_fit_spacecraft['angle'].angle.rad,
          polarization_fit_spacecraft['angle uncertainty'].rad)

    #assert np.allclose([polarization_fit_spacecraft['fraction'],
    #                    polarization_fit_spacecraft['fraction uncertainty'],
    #                    polarization_fit_spacecraft['angle'].angle.rad,
    #                    polarization_fit_spacecraft['angle uncertainty'].rad],
    #                   [0.8248301194006322, 0.7889544546660269,
    #                    1.5645814961205469, 0.5143016570410734],
    #                   atol=[0.2, 0.1, 0.2, 0.1])

    # test pseudostokes plotting code, including background
    polarization_spacecraft.plot_pseudostokes()

def test_stokes_icrs_fit():

    polarization_icrs = PolarizationStokes(source_direction.transform_to('galactic'),
                                           spectrum, bin_edges,
                                           unbinned_data,
                                           sc_orientation, response_path,
                                           background=background,
                                           show_plots=True)

    polarization_fit_icrs = polarization_icrs.fit(show_plots=True)

    print(polarization_fit_icrs['fraction'],
          polarization_fit_icrs['fraction uncertainty'],
          polarization_fit_icrs['angle'].angle.rad,
          polarization_fit_icrs['angle uncertainty'].rad)

    #assert np.allclose([polarization_fit_icrs['fraction'],
    #                    polarization_fit_icrs['fraction uncertainty'],
    #                    polarization_fit_icrs['angle'].angle.rad,
    #                    polarization_fit_icrs['angle uncertainty'].rad],
    #                   [1.496745801986812, 0.8751758133432178,
    #                    1.84399578798795, 0.3812557749920544],
    #                   atol=[0.2, 0.1, 0.2, 0.1])

    # test pseudostokes plotting code, including background
    polarization_icrs.plot_pseudostokes()

def test_stokes_nobg_fit():

    polarization = PolarizationStokes(source_direction, spectrum, bin_edges, unbinned_data,
                                      sc_orientation, response_path, background=None,
                                      show_plots=True)

    average_mu = polarization._mu100['mu']
    mdp99 = polarization._mdp99

    assert np.allclose([average_mu, mdp99], [0.19, 0.22], atol=[0.1, 0.1])

    polarization_fit = polarization.fit(show_plots=True)

    assert np.allclose([polarization_fit['fraction'],
                        polarization_fit['fraction uncertainty'],
                        polarization_fit['angle'].angle.deg,
                        polarization_fit['angle uncertainty'].deg],
                       [1.8181606920477378, 0.06885360774325616,
                        81.95528281435703, 1.1209284368621406],
                       atol=[0.05, 0.1, 10, 1])

    # test pseudostokes plotting code
    polarization.plot_pseudostokes()

    # test rotate_photons_to_x_axis
    test_pd, test_pa = 0.8, 90
    test_q, test_u = polarization.rotate_points_to_x_axis(test_pd, np.radians(test_pa))
    assert np.allclose([test_q, test_u], [-0.8, 0])
