"""
Response-configured, arbitrary-bin true-energy SED model for COSI/3ML.

``BinnedSED`` assigns one independent differential normalization to each
selected response true-energy bin. Within a bin, the energy dependence is a
frozen copy of a user-provided astromodels spectral shape, renormalized to
unity at the bin pivot.
"""

import copy

import numpy as np
import astropy.units as u

from astromodels import Powerlaw
from astromodels.functions.function import Function1D, FunctionMeta


__all__ = ["BinnedSED"]


_BINNED_SED_CLASSES = {}


class BinnedSED(Function1D):
    """
    Base type for a response-configured arbitrary-bin SED.

    Do not instantiate this class directly. Use ``BinnedSED.from_response``;
    the number of astromodels parameters must be known when the concrete
    Function1D class is created.
    """

    _n_sed_bins = None

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "BinnedSED must be created with BinnedSED.from_response(response, ...)."
        )

    @classmethod
    def from_response(
        cls,
        response,
        spectral_shape=None,
        ei_bin_indices=None,
        initial_fluxes=None,
        default_initial_flux=1e-8,
    ):
        """
        Create and configure a binned SED directly from response Ei bins.

        Parameters
        ----------
        response : ExtendedSourceResponse-like
            Object exposing ``response.axes[\"Ei\"]`` with ``edges`` and
            ``nbins``.
        spectral_shape : astromodels Function1D-like, optional
            Spectral shape to retain inside every bin. A deep copy is stored
            and all of its parameters are frozen. Its absolute normalization
            cancels when the shape is normalized at each bin pivot. If omitted,
            a power law with index -2 is used.
        ei_bin_indices : iterable of int, optional
            Contiguous increasing response Ei-bin indices to use. If omitted,
            all response Ei bins are used.
        initial_fluxes : array-like or Quantity, optional
            Initial K_i values, one per selected response bin. If omitted, the
            model defaults are retained.
        default_initial_flux : float, optional
            Positive fallback replacing any non-finite or non-positive supplied
            initial flux. Default is 1e-8.

        Returns
        -------
        BinnedSED
            A concrete astromodels Function1D whose number of K_i and E_i
            parameters matches the selected response bins.
        """

        if spectral_shape is None:
            spectral_shape = Powerlaw()
            spectral_shape.index.value = -2.0
        elif not callable(spectral_shape) or not hasattr(
            spectral_shape, "parameters"
        ):
            raise TypeError(
                "spectral_shape must be a callable astromodels Function1D-like "
                "object exposing parameters."
            )

        ei_axis = response.axes["Ei"]

        if ei_bin_indices is None:
            bins = np.arange(ei_axis.nbins, dtype=int)
        else:
            bins = np.asarray(list(ei_bin_indices), dtype=int)

        if bins.size == 0:
            raise ValueError("At least one response Ei bin must be selected.")

        if not np.all(np.diff(bins) == 1):
            raise ValueError(
                "Selected response Ei bins must be contiguous and increasing."
            )

        if bins[0] < 0 or bins[-1] >= ei_axis.nbins:
            raise IndexError(
                f"Selected Ei bins must lie in [0, {ei_axis.nbins - 1}]."
            )

        frozen_shape = copy.deepcopy(spectral_shape)
        for parameter in frozen_shape.parameters.values():
            parameter.free = False

        n_bins = int(bins.size)
        concrete_class = _get_binned_sed_class(n_bins, frozen_shape)
        spectrum = concrete_class()
        spectrum._spectral_shape = copy.deepcopy(frozen_shape)

        edges = ei_axis.edges

        # COSI response true energies are conventionally keV. When the response
        # carries explicit units, convert to keV so the numerical parameter
        # values are ready for the standard astromodels spectral energy unit.
        if isinstance(edges, u.Quantity):
            selected_edges = np.asarray(
                edges[bins[0] : bins[-1] + 2].to_value(u.keV),
                dtype=float,
            )
        else:
            selected_edges = np.asarray(
                edges[bins[0] : bins[-1] + 2],
                dtype=float,
            )

        if np.any(np.diff(selected_edges) <= 0.0):
            raise ValueError("Selected response Ei edges are not strictly increasing.")

        for i, edge in enumerate(selected_edges):
            par = getattr(spectrum, f"E{i}")
            par.value = float(edge)
            par.free = False

        if not np.isfinite(default_initial_flux) or default_initial_flux <= 0.0:
            raise ValueError("default_initial_flux must be finite and positive.")

        if initial_fluxes is not None:
            if isinstance(initial_fluxes, u.Quantity):
                initial_fluxes = initial_fluxes.value

            initial_fluxes = np.asarray(initial_fluxes, dtype=float)

            if initial_fluxes.size != n_bins:
                raise ValueError(
                    f"initial_fluxes must contain exactly {n_bins} values."
                )

            initial_fluxes = np.where(
                np.isfinite(initial_fluxes) & (initial_fluxes > 0.0),
                initial_fluxes,
                float(default_initial_flux),
            )

            for i, flux in enumerate(initial_fluxes):
                getattr(spectrum, f"K{i}").value = float(flux)

        for i in range(n_bins):
            getattr(spectrum, f"K{i}").free = True

        # Convenience metadata used by diagnostics and notebook output.
        spectrum._cosipy_ei_bin_indices = tuple(int(i) for i in bins)
        spectrum._validate_spectral_shape()
        return spectrum

    @property
    def n_bins(self):
        """Number of SED bins in this concrete spectrum."""
        return int(self._n_sed_bins)

    @property
    def bin_edges(self):
        """Current numerical SED energy edges."""
        return np.asarray(
            [getattr(self, f"E{i}").value for i in range(self.n_bins + 1)],
            dtype=float,
        )

    @property
    def pivots(self):
        """Geometric-center pivot energy of each SED bin."""
        edges = self.bin_edges
        return np.sqrt(edges[:-1] * edges[1:])

    @property
    def normalizations(self):
        """Tuple containing K0 ... K(N-1) Parameter objects."""
        return tuple(getattr(self, f"K{i}") for i in range(self.n_bins))

    @property
    def spectral_shape(self):
        """Frozen copy of the user-provided within-bin spectral shape."""
        try:
            return object.__getattribute__(self, "_spectral_shape")
        except AttributeError:
            # 3ML clones fitted astromodels models by serializing their public
            # parameters and reconstructing the Function1D. Private instance
            # attributes are not serialized, so restore the frozen shape from
            # the generated concrete class in that reconstructed object.
            spectral_shape = copy.deepcopy(type(self)._spectral_shape_template)
            object.__setattr__(self, "_spectral_shape", spectral_shape)
            return spectral_shape

    def _set_units_impl(self, x_unit, y_unit):
        for i in range(self.n_bins + 1):
            getattr(self, f"E{i}").unit = x_unit

        for parameter in self.normalizations:
            parameter.unit = y_unit

        self.spectral_shape.set_units(x_unit, y_unit)

    @staticmethod
    def _value_in_unit(value, unit):
        if isinstance(value, u.Quantity):
            return value.to_value(unit)
        return np.asarray(value, dtype=float)

    def _shape_values(self, energy):
        values = self.spectral_shape(energy)
        if isinstance(values, u.Quantity):
            values = values.value
        return np.asarray(values, dtype=float)

    def _validate_spectral_shape(self):
        pivot_values = self._shape_values(self.pivots)
        if pivot_values.shape != self.pivots.shape:
            pivot_values = np.broadcast_to(pivot_values, self.pivots.shape)
        if np.any(~np.isfinite(pivot_values)) or np.any(pivot_values <= 0.0):
            raise ValueError(
                "spectral_shape must be finite and strictly positive at every "
                "selected SED-bin pivot."
            )

    def _evaluate_impl(self, x, kvals_in, edges_in):
        x_has_units = isinstance(x, u.Quantity)

        if x_has_units:
            xv = np.asarray(x.to_value(self.x_unit), dtype=float)
            edges = np.asarray(
                [self._value_in_unit(edge, self.x_unit) for edge in edges_in],
                dtype=float,
            )
            kvals = np.asarray(
                [self._value_in_unit(k, self.y_unit) for k in kvals_in],
                dtype=float,
            )
            shape_energy = np.atleast_1d(xv) * self.x_unit
            shape_pivots = self.pivots * self.x_unit
        else:
            xv = np.asarray(x, dtype=float)
            edges = np.asarray(edges_in, dtype=float)
            kvals = np.asarray(kvals_in, dtype=float)
            shape_energy = np.atleast_1d(xv)
            shape_pivots = self.pivots

        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("BinnedSED energy edges must be strictly increasing.")

        scalar_input = xv.ndim == 0
        x_eval = np.atleast_1d(xv)
        shape_values = self._shape_values(shape_energy)
        pivot_values = self._shape_values(shape_pivots)
        if np.any(~np.isfinite(pivot_values)) or np.any(pivot_values <= 0.0):
            raise ValueError(
                "The spectral shape is not finite and positive at all pivots."
            )
        flux = np.zeros_like(x_eval, dtype=float)

        for i in range(self.n_bins):
            elo = edges[i]
            ehi = edges[i + 1]

            if i < self.n_bins - 1:
                mask = (x_eval >= elo) & (x_eval < ehi)
            else:
                mask = (x_eval >= elo) & (x_eval <= ehi)

            if np.any(mask):
                flux[mask] = kvals[i] * shape_values[mask] / pivot_values[i]

        result = flux[0] if scalar_input else flux

        if x_has_units:
            return result * self.y_unit

        return result

    def integral(self, a, b):
        """
        Integral between two numerical energy boundaries.

        This follows the astromodels ``Function1D.integral`` convention and
        returns a plain numerical value. Use ``Function1D.integrate`` for
        Quantity boundaries and a unit-bearing result.
        """

        if isinstance(a, u.Quantity):
            a = a.to_value(self.x_unit)
        if isinstance(b, u.Quantity):
            b = b.to_value(self.x_unit)

        av = float(a)
        bv = float(b)

        if bv < av:
            return -self.integral(bv, av)

        edges = self.bin_edges
        kvals = np.asarray([par.value for par in self.normalizations], dtype=float)
        pivot_values = self._shape_values(self.pivots)

        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("BinnedSED energy edges must be strictly increasing.")

        # Local import avoids a module-level circular dependency.
        from cosipy.response.integrals import get_integral_values

        total = 0.0
        for i in range(self.n_bins):
            lo = max(av, edges[i])
            hi = min(bv, edges[i + 1])

            if hi <= lo:
                continue

            shape_integral = get_integral_values(
                self.spectral_shape, np.asarray([lo, hi], dtype=float)
            )[0]
            total += kvals[i] * shape_integral / pivot_values[i]

        return float(total)


def _make_function_doc(n_bins):
    lines = [
        "description :",
        f"    Frozen-shape SED with {n_bins} response-defined true-energy bins.",
        "parameters :",
    ]

    for i in range(n_bins):
        lines.extend(
            [
                f"    K{i} :",
                f"        desc : Differential normalization in true-energy bin {i}",
                "        initial value : 1e-6",
            ]
        )
        if i == 0:
            lines.append("        is_normalization : True")
        lines.extend(
            [
                "        min : 0",
                "        max : 1e-2",
                "        delta : 1e-7",
            ]
        )

    for i in range(n_bins + 1):
        if i == 0:
            desc = "Lower edge of SED bin 0"
        elif i == n_bins:
            desc = f"Upper edge of SED bin {n_bins - 1}"
        else:
            desc = f"Edge between SED bins {i - 1} and {i}"

        lines.extend(
            [
                f"    E{i} :",
                f"        desc : {desc}",
                f"        initial value : {i + 1}",
                "        fix : yes",
            ]
        )

    return "\n".join(lines)


def _make_evaluate(n_bins):
    k_names = [f"K{i}" for i in range(n_bins)]
    e_names = [f"E{i}" for i in range(n_bins + 1)]
    parameters = k_names + e_names

    source = (
        f"def evaluate(self, x, {', '.join(parameters)}):\n"
        f"    return self._evaluate_impl(x, [{', '.join(k_names)}], "
        f"[{', '.join(e_names)}])\n"
    )

    namespace = {}
    exec(source, {}, namespace)
    return namespace["evaluate"]


def _set_units(self, x_unit, y_unit):
    self._set_units_impl(x_unit, y_unit)


def _spectral_shape_cache_key(spectral_shape):
    """Return a stable in-process key for a frozen astromodels shape."""

    return (
        type(spectral_shape).__module__,
        type(spectral_shape).__name__,
        repr(spectral_shape.to_dict()),
    )


def _get_binned_sed_class(n_bins, spectral_shape):
    n_bins = int(n_bins)

    if n_bins < 1:
        raise ValueError("BinnedSED requires at least one bin.")

    cache_key = (n_bins, _spectral_shape_cache_key(spectral_shape))

    if cache_key not in _BINNED_SED_CLASSES:
        class_name = f"BinnedSED_{n_bins}_{len(_BINNED_SED_CLASSES)}"
        namespace = {
            "__doc__": _make_function_doc(n_bins),
            "__module__": __name__,
            "_n_sed_bins": n_bins,
            "_spectral_shape_template": copy.deepcopy(spectral_shape),
            "evaluate": _make_evaluate(n_bins),
            "_set_units": _set_units,
        }

        concrete_class = FunctionMeta(class_name, (BinnedSED,), namespace)
        _BINNED_SED_CLASSES[cache_key] = concrete_class

        # Make the generated class discoverable in this module after creation.
        globals()[class_name] = concrete_class

    return _BINNED_SED_CLASSES[cache_key]
