"""
Response-configured, arbitrary-bin true-energy SED model for COSI/3ML.

The public entry point is ``BinnedSED.from_response``.  It creates an
astromodels ``Function1D`` with one independent normalization K_i for each
selected response true-energy bin.  The selected bins must be contiguous.

Within bin i,

    dN/dE = K_i * (E / E_piv,i)**index

with E_piv,i = sqrt(E_i * E_{i+1}).  The bin edges and common local index are
fixed, while the K_i values are free fit parameters.
"""

import numpy as np
import astropy.units as u

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
        ei_bin_indices=None,
        initial_fluxes=None,
        index=-2.0,
        default_initial_flux=1e-8,
    ):
        """
        Create and configure a binned SED directly from response Ei bins.

        Parameters
        ----------
        response : ExtendedSourceResponse-like
            Object exposing ``response.axes[\"Ei\"]`` with ``edges`` and
            ``nbins``.
        ei_bin_indices : iterable of int, optional
            Contiguous increasing response Ei-bin indices to use. If omitted,
            all response Ei bins are used.
        initial_fluxes : array-like or Quantity, optional
            Initial K_i values, one per selected response bin. If omitted, the
            model defaults are retained.
        index : float, optional
            Fixed local power-law index in every SED bin. Default is -2.
        default_initial_flux : float, optional
            Positive fallback replacing any non-finite or non-positive supplied
            initial flux. Default is 1e-8.

        Returns
        -------
        BinnedSED
            A concrete astromodels Function1D whose number of K_i and E_i
            parameters matches the selected response bins.
        """

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

        n_bins = int(bins.size)
        concrete_class = _get_binned_sed_class(n_bins)
        spectrum = concrete_class()

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

        spectrum.index.value = float(index)
        spectrum.index.free = False

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

    def _set_units_impl(self, x_unit, y_unit):
        for i in range(self.n_bins + 1):
            getattr(self, f"E{i}").unit = x_unit

        for i in range(self.n_bins):
            getattr(self, f"K{i}").unit = y_unit

        self.index.unit = u.dimensionless_unscaled

    @staticmethod
    def _value_in_unit(value, unit):
        if isinstance(value, u.Quantity):
            return value.to_value(unit)
        return np.asarray(value, dtype=float)

    def _evaluate_impl(self, x, kvals_in, edges_in, index):
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
        else:
            xv = np.asarray(x, dtype=float)
            edges = np.asarray(edges_in, dtype=float)
            kvals = np.asarray(kvals_in, dtype=float)

        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("BinnedSED energy edges must be strictly increasing.")

        index_value = float(getattr(index, "value", index))
        scalar_input = xv.ndim == 0
        x_eval = np.atleast_1d(xv)
        flux = np.zeros_like(x_eval, dtype=float)

        for i in range(self.n_bins):
            elo = edges[i]
            ehi = edges[i + 1]
            epiv = np.sqrt(elo * ehi)

            if i < self.n_bins - 1:
                mask = (x_eval >= elo) & (x_eval < ehi)
            else:
                mask = (x_eval >= elo) & (x_eval <= ehi)

            if np.any(mask):
                flux[mask] = kvals[i] * np.power(
                    x_eval[mask] / epiv,
                    index_value,
                )

        result = flux[0] if scalar_input else flux

        if x_has_units:
            return result * self.y_unit

        return result

    def integral(self, a, b):
        """
        Exact integral between two numerical energy boundaries.

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
        idx = float(self.index.value)

        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("BinnedSED energy edges must be strictly increasing.")

        total = 0.0

        for i in range(self.n_bins):
            lo = max(av, edges[i])
            hi = min(bv, edges[i + 1])

            if hi <= lo:
                continue

            epiv = np.sqrt(edges[i] * edges[i + 1])

            if np.isclose(idx, -1.0):
                integ = kvals[i] * epiv * np.log(hi / lo)
            else:
                integ = (
                    kvals[i]
                    * epiv
                    / (idx + 1.0)
                    * (
                        np.power(hi / epiv, idx + 1.0)
                        - np.power(lo / epiv, idx + 1.0)
                    )
                )

            total += integ

        return float(total)


def _make_function_doc(n_bins):
    lines = [
        "description :",
        f"    Piecewise power-law SED with {n_bins} response-defined true-energy bins.",
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

    lines.extend(
        [
            "    index :",
            "        desc : Local power-law index in every SED bin",
            "        initial value : -2",
            "        min : -10",
            "        max : 10",
            "        fix : yes",
        ]
    )

    return "\n".join(lines)


def _make_evaluate(n_bins):
    k_names = [f"K{i}" for i in range(n_bins)]
    e_names = [f"E{i}" for i in range(n_bins + 1)]
    parameters = k_names + e_names + ["index"]

    source = (
        f"def evaluate(self, x, {', '.join(parameters)}):\n"
        f"    return self._evaluate_impl(x, [{', '.join(k_names)}], "
        f"[{', '.join(e_names)}], index)\n"
    )

    namespace = {}
    exec(source, {}, namespace)
    return namespace["evaluate"]


def _set_units(self, x_unit, y_unit):
    self._set_units_impl(x_unit, y_unit)


def _get_binned_sed_class(n_bins):
    n_bins = int(n_bins)

    if n_bins < 1:
        raise ValueError("BinnedSED requires at least one bin.")

    if n_bins not in _BINNED_SED_CLASSES:
        class_name = f"BinnedSED_{n_bins}"
        namespace = {
            "__doc__": _make_function_doc(n_bins),
            "__module__": __name__,
            "_n_sed_bins": n_bins,
            "evaluate": _make_evaluate(n_bins),
            "_set_units": _set_units,
        }

        concrete_class = FunctionMeta(class_name, (BinnedSED,), namespace)
        _BINNED_SED_CLASSES[n_bins] = concrete_class

        # Make the generated class discoverable in this module after creation.
        globals()[class_name] = concrete_class

    return _BINNED_SED_CLASSES[n_bins]
