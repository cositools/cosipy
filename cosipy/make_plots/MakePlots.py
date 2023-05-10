# Imports:
import matplotlib.pyplot as plt

class MakePlots:

    def make_basic_plot(self, x, y, plt_scale='loglog', savefig='plot.pdf',\
        x_error=[], plot_kwargs={}, fig_kwargs={}):

        """
        Make a basic plot.

        Inputs:
        x [list or array]: x-axis data to be plotted
        y [list or array]: y-axis data to be plotted

        Optional:
        plt_scale: scale of axes: loglog, semilogx, or semilogy.
        x_error [list or array]: x error bars
        savefig [str]: Name of saved figure
        plot_kwargs [dict]: pass any kwargs to plt.plot()
        fig_kwargs [dict]: pass any kwargs to plt.gca().set()
        """
        
        # Setup figure:
        ax = plt.gca()

        # Main plot:
        if plt_scale == "loglog":
            plt.loglog(x, y, **plot_kwargs)
        if plt_scale == "semilogx":
            plt.semilogx(x, y, **plot_kwargs)
        if plt_scale == "semilogy":
            plt.semilogy(x, y, **plot_kwargs)

        # Include x error bars:
        if len(x_error) != 0:
            
            # Remove label if defined:
            if "label" in plot_kwargs.keys():
                plot_kwargs["label"] = "_nolabel_"

            plt.errorbar(x, y, xerr=x_error, **plot_kwargs)

        # axes and labels:
        plt.grid(color="grey",alpha=0.3,ls="--")
        ax.set(**fig_kwargs)
        if "label" in plot_kwargs.keys():
            plt.legend(loc=1,frameon=True)
        
        # Save and show:
        plt.savefig(savefig)
        plt.show()
        plt.close()

        return
