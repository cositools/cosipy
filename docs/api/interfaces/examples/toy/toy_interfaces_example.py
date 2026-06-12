from toy_implementations import *

from astromodels import Parameter
import astropy.units as u
from astropy.time import Time
from cosipy.event_selection.time_selection import TimeSelector
from cosipy.interfaces.expectation_interface import SumExpectationDensity

from cosipy.statistics import PoissonLikelihood, UnbinnedLikelihood

from cosipy.interfaces import ThreeMLPluginInterface
from histpy import Axis, Histogram
import numpy as np

from threeML import Constant, PointSource, Model, JointLikelihood, DataList

from matplotlib import pyplot as plt


def main():

    # This axis is user for binning the data in the binned analysis case
    # The unbinned analysis also uses to lower and upper limits, as well as for plotting
    toy_axis = Axis(np.linspace(-5, 5), label='x')

    # Some options
    unbinned = False     # Binned=False or unbinned=True
    plot = True         # Plots the fit
    use_signal = True   # False = bkg-only
    use_bkg = True      # False = signal-only

    # This simulates a stream of events. It can come from a file or some other source
    # ToyEventDataStream and ToyEventData could have been simplified into a single
    # class, but I wanted to exercise the case of a consumable stream, which is
    # cached by ToyEventData and used in the rest of the analysis.
    # The event have an 'x' value and time.
    # For the signal, the 'x' values are randomly drawn from a standard normal distribution
    # For the background, the 'x' value are randomly drawn from a uniform distribution
    # The timestamps are randomly drawn from a uniform distribution in both cases.
    # All the events are time-sorted.
    data_loader = ToyEventDataStream(nevents_signal= 1000 if use_signal else 0,
                                     nevents_bkg= 1000 if use_bkg else 0,
                                     min_value=toy_axis.lo_lim,
                                     max_value=toy_axis.hi_lim,
                                     tstart=Time("2000-01-01T00:00:00"),
                                     tstop=Time("2000-01-02T00:00:00"))

    # Make a selection. A simple time selection in this case
    # TimeSelector assumed the events are time-sorted and will stop the stream
    # of events once tstop is reached
    tstart = Time("2000-01-01T01:00:00")
    tstop = Time("2000-01-01T10:00:00")
    duration = tstop - tstart
    selector = TimeSelector(tstart = tstart, tstop = tstop)

    event_data = ToyEventData(data_loader, selector=selector)

    # Data binning true the interface fill() method
    binned_data = None
    if plot or not unbinned:
        binned_data = ToyBinnedData(Histogram(toy_axis))
        binned_data.fill(event_data)

    # This is the expectation from a single source, which is just the standard normal
    # distribution
    # This class handles both the binned and the unbinned case.
    if unbinned:
        psr = ToyPointSourceResponse(data = event_data, duration = duration)
    else:
        psr = ToyPointSourceResponse(data=binned_data, duration=duration)

    # This combines the expectation from multiple
    model_folding = ToyModelFolding(data = event_data, psr = psr)

    if use_bkg:
        # The expectation from background, which is flat
        # This class handles both the binned and the unbinned case
        bkg = ToyBkg(data = event_data, duration = duration, axis = toy_axis)
    else:
        bkg = None

    # Source model
    # Since this is a toy model with no position or energy dependence,
    # we'll just use the normalization K value and ignore the units
    # The default units are 1 / (keV s cm2), which make sure for an astrophysical
    # source, but for this toy model.
    spectrum = Constant()

    if use_signal:
        spectrum.k.value = .01
    else:
        spectrum.k.value = 0
        spectrum.k.free = False

    spectrum.k.min_value = 0

    source = PointSource("arbitrary_source_name",
                             l=0, b=0,  # Doesn't matter
                             spectral_shape=spectrum)

    model = Model(source)

    # Set the likelihood function we'll use
    if unbinned:
        expectation_density = SumExpectationDensity(model_folding, bkg)
        like_fun = UnbinnedLikelihood(expectation_density)
    else:
        like_fun = PoissonLikelihood(binned_data, model_folding, bkg)


    # Initiate the 3ML plugin
    # This plugin will internally call
    # response.set_model() and bkg.set_parameter()
    # which will cause the like_fun result to change on each call
    cosi = ThreeMLPluginInterface('cosi',
                                  like_fun,
                                  response = model_folding,
                                  bkg = bkg)

    # Before the fit, you can set the background parameters initial values, bounds, etc.
    # This is passed to the minimizer.
    # The source model parameters were already set above
    if bkg is not None:
        cosi.bkg_parameter['norm'] = Parameter("norm",  # background parameter
                                      1,  # initial value of parameter
                                      unit = u.Hz,
                                      min_value=0,  # minimum value of parameter
                                      max_value=1,  # maximum value of parameter
                                      delta=0.001,  # initial step used by fitting engine
                                      free = True)

    # Fit
    plugins = DataList(cosi) # Each instrument or data set
    like = JointLikelihood(model, plugins) # Everything connects here

    like.fit()
    print(like.minimizer)

    # Plot results
    if plot:

        fig, ax = plt.subplots()


        if unbinned:
            # Divide by bin width to plot the density
            (binned_data.data/toy_axis.widths).plot(ax)

            # Get the expectation density from the fitted result for each event
            expectation_density_list = np.fromiter(expectation_density.expectation_density(), dtype=float)
            ax.scatter(event_data.x, expectation_density_list, s=1, color='green')

            ax.set_ylabel("Counts density")
        else:
            binned_data.data.plot(ax)
            expectation = model_folding.expectation(binned_data.axes)

            if bkg is not None:
                expectation = expectation + bkg.expectation(binned_data.axes)

            expectation.plot(ax)

            ax.set_ylabel("Counts")

        plt.show()

if __name__ == "__main__":

    import cProfile
    cProfile.run('main()', filename = "prof_toy.prof")
    exit()

    main()