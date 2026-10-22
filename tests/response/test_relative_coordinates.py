import numpy as np

from cosipy.polarization import StereographicConvention
from cosipy.response.relative_coordinates import RelativeCDSCoordinates


def _unit_vectors(n, seed=0):
    v = np.random.default_rng(seed).normal(size=(3, n))
    return v / np.linalg.norm(v, axis=0)


def test_to_relative_is_finite_at_the_source_direction_and_its_opposite():
    """The dot product of a unit vector with itself can round slightly above 1."""

    source = _unit_vectors(10000)
    relcoords = RelativeCDSCoordinates(source, StereographicConvention())

    phi, az = relcoords.to_relative(source)
    phi_opposite, _ = relcoords.to_relative(-source)

    assert np.all(np.isfinite(phi.rad)) and np.all(np.isfinite(az.rad))
    np.testing.assert_allclose(phi.rad, 0, atol=1e-7)
    np.testing.assert_allclose(phi_opposite.rad, np.pi, atol=1e-7)


def test_to_relative_inverts_to_cds():
    source = _unit_vectors(1000, seed=1)
    relcoords = RelativeCDSCoordinates(source, StereographicConvention())
    rng = np.random.default_rng(2)
    phi = rng.uniform(0.01, np.pi - 0.01, 1000)
    az = rng.uniform(-np.pi, np.pi, 1000)

    phi_back, az_back = relcoords.to_relative(np.asarray(relcoords.to_cds(phi, az)))

    np.testing.assert_allclose(phi_back.rad, phi, atol=1e-10)
    np.testing.assert_allclose(np.angle(np.exp(1j * (az_back.rad - az))), 0, atol=1e-10)
