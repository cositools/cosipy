"""
Response-configured, arbitrary-bin true-energy SED model for COSI/3ML.

``BinnedSED`` assigns one independent differential normalization to each
selected response true-energy bin. Within a bin, the energy dependence is a
copy of a user-provided astromodels spectral shape, renormalized to unity at
the bin pivot. Shape parameters are fixed by default, but may be independently
freed in selected bins through ``bin_shape_parameters``.
"""

import copy
import json
import operator

import numpy as np
import astropy.units as u

from astromodels import Powerlaw
from astromodels.core.parameter_transformation import LogarithmicTransformation
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
            Reference spectral shape for every bin. Independent copies of its
            parameters are registered with astromodels, initially all fixed,
            regardless of the input free flags. Use ``bin_shape_parameters(i)``
            to change their values or free selected parameters after creation.
            Its overall normalization cancels at each bin pivot. If omitted,
            a power law with fixed index -2 is used.
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
            parameters matches the selected response bins, with independent
            shape parameters for each bin.

        Examples
        --------
        Only bin normalizations are free initially. To also fit the index
        in local SED bin 2 (the third selected response bin)::

            sed = BinnedSED.from_response(response, Powerlaw())
            sed.bin_shape_parameters(2)["index"].free = True

        The equivalent astromodels parameter is ``sed.shape2_index``. Other bins
        retain their fixed cutoff energies.

        Do not free redundant parameters that cancel in the shape/pivot ratio
        (e.g. the standalone power law's K or piv).
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
        snapshot_values = {
            name: parameter.value for name, parameter in frozen_shape.parameters.items()
        }
        for name, parameter in frozen_shape.parameters.items():
            if parameter.has_auxiliary_variable:
                parameter.remove_auxiliary_variable()
                parameter.value = snapshot_values[name]
            parameter.free = False

        n_bins = int(bins.size)
        concrete_class = _get_binned_sed_class(n_bins, frozen_shape)
        spectrum = concrete_class()
        spectrum._spectral_shape = copy.deepcopy(frozen_shape)
        for i in range(n_bins):
            for name, parameter in frozen_shape.parameters.items():
                if parameter.has_prior():
                    spectrum.bin_shape_parameters(i)[name].prior = copy.deepcopy(
                        parameter.prior
                    )

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
        """Reference shape, not the possibly modified shape of any single bin.

        Use ``bin_shape_parameters`` to edit the SED and ``bin_spectral_shape``
        to inspect a bin's current shape. This reference is retained for
        compatibility with fixed-shape consumers.
        """
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

    def bin_shape_parameters(self, bin_index):
        """Return a name-to-Parameter mapping for one local SED bin.

        Indices are zero-based within the selected SED bins, not absolute
        response Ei indices. The returned objects are registered astromodels
        parameters: changing their values, bounds or ``free`` flags affects
        fitting. All are fixed initially. For example::

            sed.bin_shape_parameters(1)["index"].free = True

        These parameters are independent of those in every other bin.
        """
        bin_index = operator.index(bin_index)
        if not 0 <= bin_index < self.n_bins:
            raise IndexError(f"SED bin index must lie in [0, {self.n_bins - 1}].")
        return {
            name: self.parameters[f"shape{bin_index}_{name}"]
            for name in type(self)._shape_parameter_names
        }

    def bin_spectral_shape(self, bin_index):
        """Return a detached snapshot of one bin's current spectral shape.

        Editing this snapshot does not edit the SED. Use
        ``bin_shape_parameters(bin_index)`` to modify the fitted parameters.
        """
        parameters = self.bin_shape_parameters(bin_index)
        shape = copy.deepcopy(self._shape_for_bin(bin_index))
        for name, parameter in parameters.items():
            shape.parameters[name].free = parameter.free
        return shape

    def _shape_for_bin(self, bin_index, values=None):
        # Private evaluation copies are reconstructed lazily after a 3ML clone.
        # The public shape{i}_* parameters carry the fitted/serialized state.
        try:
            shapes = object.__getattribute__(self, "_bin_spectral_shapes")
        except AttributeError:
            shapes = [copy.deepcopy(self.spectral_shape) for _ in range(self.n_bins)]
            object.__setattr__(self, "_bin_spectral_shapes", shapes)
        shape = shapes[bin_index]
        parameters = self.bin_shape_parameters(bin_index)
        if values is None:
            values = [parameter.value for parameter in parameters.values()]
        for (name, parameter), value in zip(parameters.items(), values):
            target = shape.parameters[name]
            value = float(self._value_in_unit(value, parameter.unit))
            # The public parameter enforces the user's bounds. Internal copies
            # must not reject values allowed after expanding those bounds.
            if target.min_value is not None and value < target.min_value:
                target.min_value = value
            if target.max_value is not None and value > target.max_value:
                target.max_value = value
            target.value = value
        return shape

    def _set_units_impl(self, x_unit, y_unit):
        for i in range(self.n_bins + 1):
            getattr(self, f"E{i}").unit = x_unit

        for parameter in self.normalizations:
            parameter.unit = y_unit

        self.spectral_shape.set_units(x_unit, y_unit)
        for i in range(self.n_bins):
            shape = self._shape_for_bin(i)
            shape.set_units(x_unit, y_unit)
            for name, parameter in self.bin_shape_parameters(i).items():
                parameter.unit = shape.parameters[name].unit

    @staticmethod
    def _value_in_unit(value, unit):
        if isinstance(value, u.Quantity):
            return value.to_value(unit)
        return np.asarray(value, dtype=float)

    def _shape_values(self, energy, shape=None):
        values = (self.spectral_shape if shape is None else shape)(energy)
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

    def _evaluate_impl(self, x, kvals_in, edges_in, shape_values_in):
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
        flux = np.zeros_like(x_eval, dtype=float)

        for i in range(self.n_bins):
            elo = edges[i]
            ehi = edges[i + 1]

            if i < self.n_bins - 1:
                mask = (x_eval >= elo) & (x_eval < ehi)
            else:
                mask = (x_eval >= elo) & (x_eval <= ehi)

            if np.any(mask):
                shape = self._shape_for_bin(i, shape_values_in[i])
                pivot_value = self._shape_values(shape_pivots[i], shape)
                if not np.isfinite(pivot_value) or pivot_value <= 0.0:
                    raise ValueError(
                        "The spectral shape must be finite and positive at the bin pivot."
                    )
                values = self._shape_values(shape_energy[mask], shape)
                flux[mask] = kvals[i] * values / pivot_value

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

            shape = self._shape_for_bin(i)
            pivot_value = self._shape_values(self.pivots[i], shape)
            if not np.isfinite(pivot_value) or pivot_value <= 0.0:
                raise ValueError(
                    "The spectral shape must be finite and positive at the bin pivot."
                )
            shape_integral = get_integral_values(
                shape, np.asarray([lo, hi], dtype=float)
            )[0]
            total += kvals[i] * shape_integral / pivot_value

        return float(total)


def _make_function_doc(n_bins, spectral_shape):
    lines = [
        "description :",
        f"    SED with {n_bins} response-defined bins and fixed-by-default shapes.",
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

    for i in range(n_bins):
        for name, parameter in spectral_shape.parameters.items():
            lines.extend(
                [
                    f"    shape{i}_{name} :",
                    f"        desc : Shape parameter {name} in local SED bin {i}",
                    f"        initial value : {float(parameter.value)!r}",
                    f"        unit : {json.dumps(str(parameter.unit))}",
                    "        fix : yes",
                ]
            )
            for key, value in (
                ("min", parameter.min_value),
                ("max", parameter.max_value),
                ("delta", parameter.delta),
            ):
                if value is not None:
                    lines.append(f"        {key} : {float(value)!r}")
            if parameter.transformation is not None:
                # astromodels currently provides the log10 transformation.
                if not isinstance(parameter.transformation, LogarithmicTransformation):
                    raise TypeError(
                        f"Unsupported transformation for spectral-shape parameter {name}."
                    )
                lines.append("        transformation : log10")

    return "\n".join(lines)


def _make_evaluate(n_bins, spectral_shape):
    k_names = [f"K{i}" for i in range(n_bins)]
    e_names = [f"E{i}" for i in range(n_bins + 1)]
    shape_names = [
        [f"shape{i}_{name}" for name in spectral_shape.parameters]
        for i in range(n_bins)
    ]
    parameters = k_names + e_names + [name for names in shape_names for name in names]
    shape_values = ", ".join(f"[{', '.join(names)}]" for names in shape_names)

    source = (
        f"def evaluate(self, x, {', '.join(parameters)}):\n"
        f"    return self._evaluate_impl(x, [{', '.join(k_names)}], "
        f"[{', '.join(e_names)}], [{shape_values}])\n"
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
        tuple(type(p.transformation) for p in spectral_shape.parameters.values()),
    )


def _get_binned_sed_class(n_bins, spectral_shape):
    n_bins = int(n_bins)

    if n_bins < 1:
        raise ValueError("BinnedSED requires at least one bin.")

    cache_key = (n_bins, _spectral_shape_cache_key(spectral_shape))

    if cache_key not in _BINNED_SED_CLASSES:
        class_name = f"BinnedSED_{n_bins}_{len(_BINNED_SED_CLASSES)}"
        namespace = {
            "__doc__": _make_function_doc(n_bins, spectral_shape),
            "__module__": __name__,
            "_n_sed_bins": n_bins,
            "_spectral_shape_template": copy.deepcopy(spectral_shape),
            "_shape_parameter_names": tuple(spectral_shape.parameters),
            "evaluate": _make_evaluate(n_bins, spectral_shape),
            "_set_units": _set_units,
        }

        concrete_class = FunctionMeta(class_name, (BinnedSED,), namespace)
        _BINNED_SED_CLASSES[cache_key] = concrete_class

        # Make the generated class discoverable in this module after creation.
        globals()[class_name] = concrete_class

    return _BINNED_SED_CLASSES[cache_key]
