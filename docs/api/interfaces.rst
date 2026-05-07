Interfaces and protocols
========================

The cosipy library has been refactored to make the components required to compute a likelihood modular and interchangeable. These components include the data (binned or event-by-event), the instrument response function (IRF), the convolution of the response with a source model, background models, and the likelihood calculation itself.

The goal is to allow users and developers to experiment with new versions of any component—without needing to modify the library core. Implementations of these components do not need to live within cosipy; they only need to comply with the relevant interface definition. This design also preserves flexibility as analysis ideas evolve over the course of the mission, and it makes it easier to switch between different approaches (for example, binned versus unbinned analyses).

This modularity is achieved by defining protocols for the interfaces between components. In practice, these are class-level contracts that specify a well-defined set of methods, including their expected inputs and outputs, that other parts of the code can rely on. The protocols describe what must be provided, but they do not prescribe how computations are performed; that is left to the specific implementation.

Below we provide an overview of the available interfaces (used interchangeably with “protocols”) and how to use them. We start from the top-most level (the likelihood) and describe all other components needed to compute it. The best place to see practical examples in action is the `Crab spectral fit tutorials  <https://github.com/cositools/cosipy/tree/develop/docs/tutorials/spectral_fits/continuum_fit/crab>`_ (binned and unbinned). Note, however, that not every part of cosipy fully uses these protocols yet; some areas still refer to specific implementations rather than a generic interface.


Here's a practical cheat sheet if you want to add/try a new:

* Instrument response function (IRF):
    - Interface: ``InstrumentResponseFunctionInterface``
    - Example: ``UnpolarizedDC3InterpolatedFarFieldInstrumentResponseFunction``
* IRF-source convolution:
    - Interface: ``ThreeMLSourceResponseInterface``
    - Examples: ``UnbinnedThreeMLPointSourceResponseTrapz``, ``BinnedThreeMLPointSourceResponse``
* Background model:
    - Interface: ``BackgroundInterface``
    - Example: ``FreeNormBinnedBackground``, ``FreeNormBackgroundInterpolatedDensityTimeTagEmCDS``
* Event selection:
    - Interface: ``EventSelectorInterface``
    - Example: ``TimeSelector``

Likelihood functions
--------------------

The ``LikelihoodInterface`` main method is ``get_log_like()``, which a promise for all implementation for result a log-likelihood. The interface, by itself, doesn't define how this will be achieved. AS a concreate example, ``PoissonLikelihood`` needs the observed counts and the expected number of counts, and internally uses the Poisson likelihood function to return the likelihood value.

Expectation interfaces
----------------------

Likelihood functions such as ``PoissonLikelihood`` require the expected  number of events --on average-- under a model that includes both signal and background contributions. This requirement is codified by the ``ExpectationInterface``. Two variants are provided: ``BinnedExpectationInterface``, which returns the expected number of counts in each bin of a binned dataset, and ``ExpectationDensityInterface``, which provides (1) the total expected number of counts and (2) the expectation density, i.e., the expected counts per unit phase space. The expectation density can be understood as the limit of the expected counts in a bin divided by the bin size as the bin size approaches zero. This form is used by the ``UnbinnedLikelihood``.

Expected counts may originate from signal sources or backgrounds.

For signal modeling, the hypothetical source is represented by the ``Source`` class from `astromodels <https://astromodels.readthedocs.io>`_. This is captured by the ``ThreeMLSourceResponseInterface``, which defines the method ``set_source(source: Source)`` for passing source properties (e.g., position, spectrum, polarization). On its own, ``ThreeMLSourceResponseInterface`` only specifies how a source is provided; it is combined with an ``ExpectationInterface`` to define an class that can accept a ``Source`` and produce expected counts. An example is ``BinnedThreeMLSourceResponseInterface`` (with a corresponding unbinned counterpart). Finally, ThreeMLModelFoldingInterface generalizes this concept to compute expectations for a collection of sources.

As you can see, protocols such as ``BinnedThreeMLSourceResponseInterface`` are intentionally abstract. This goal of this design is to constrain implementations as little as possible while still providing a reliable contract for the rest of the code. In practice, an implementation of ``BinnedThreeMLSourceResponseInterface`` will typically compute the expected number of counts by convolving ---i.e. integrate over the relevant dimensions such as time, energy, and direction--- an instrument response function (IRF) with a source model (e.g., spectrum, light curve, and/or morphology). However, the IRF itself is defined by a separate protocol (see below).

For backgrounds, we define ``BackgroundInterface``. It is intentionally generic and only requires the ability to pass parameter values via ``set_parameters()``. These parameters are completely user-defined: the interface does not assign them any specific meaning. A common concrete example is a single background normalization, but more complex cases are also supported —for example, multiple background components with independent normalizations, or non-linear parameters such as an exponential index, or even the input to a complex simulation. In a fit, these typically act as nuisance parameters. As with the signal-side expectation, ``BackgroundInterface`` is combined with an ``ExpectationInterface`` to define a complete contract for the required inputs and outputs (i.e., an object that can accept background parameters and return the corresponding expected counts).

Instrument Response Functions
-----------------------------

The (far-field) instrument response function (IRF) is responsible for providing: (1) the effective area as a function of photon energy and incoming direction (in spacecraft coordinates), and (2) the probability of obtaining a given set of measurements when a photon is detected.
These outputs are defined by ``FarFieldInstrumentResponseFunctionInterface`` via the methods ``effective_area_cm()`` and ``event_probability()``, respectively. For binned analyses, ``BinnedInstrumentResponseInterface`` provides ``differential_effective_area``, i.e., the product of ``effective_area_cm`` and ``event_probability`` integrated over each bin of a binned dataset. The interfaces definitions also allow for a near-field IRF, although it has not been developed yet.

For a Compton telescope such as COSI, typical measurements include the reconstructed energy, the Compton scattering angle, and the scattering direction. The definition of the measurement space is intentionally not part of the IRF interfaces; instead, it is delegated to the ``EventDataInterface`` (see below). Each IRF implementation must declare which ``EventDataInterface`` subclass it can handle, including any derived subclasses.

The IRF also needs to know how it will receive the photon properties being queried (e.g., true energy, direction, and polarization). This is handled by the ``PhotonInterface`` (see below).

Data Interfaces
---------------

The data interfaces are used as a medium to specify measurements and counts independently of their origin or file format. There are two types: binned data and event (unbinned) data.

The ``BinnedData`` interface is currently a thin wrapper around ``histpy``’s ``Histogram``, which contains both the axes for measured quantities and the observed counts (the histogram contents).

There are multiple ``EventDataInterface`` derivatives, each specifying a measured value through a property. For example, ``EventDataWithEnergyInterface`` provides the property ``energy_keV``, and ``EventDataWithScatteringAngleInterface`` provides ``scattering_angle_rad``. These properties return iterables, with one entry per event. Their names are descriptive and include the unit. For convenience, properties such as ``energy`` (returning an ``astropy`` ``Quantity``) are also provided, but they are typically slower and should be avoided in performance-critical code. ``EventDataInterface`` types can be combined by creating classes that inherit from multiple interfaces, e.g. ``ComptonDataSpaceInSCFrameEventDataInterface``.

In addition, all ``EventDataInterface`` implementations can return an iterable of ``EventInterface`` objects. ``EventInterface`` is similar to ``EventDataInterface``, except that it holds the values for a single event.

Photon Interfaces
-----------------

``PhotonInterface`` and ``PhotonListInterface`` work in the same way as ``EventInterface`` and ``EventDataInterface``, respectively. However, their scope is more limited, since a photon’s direction, energy, and polarization are sufficient to fully specify its state. In the future, this may be extended to include the photon origin location to support near-field analyses.

Event selection
---------------

The ``EventSelectionInterface`` defines a ``select()`` method, which takes either an ``EventDataInterface`` or an ``EventInterface`` object and returns True or False to indicate whether it should be selected. Implementations should specify which ``EventInterface`` type they support; all subclasses of that type are automatically supported. For example, ``TimeSelector`` supports ``TimetagEventInterface`` and therefore also any derived interface such as ``TimeTagEmCDSEventInSCFrameInterface``.












