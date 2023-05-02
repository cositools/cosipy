# Imports:
import matplotlib.pyplot as plt

class MakePlots:

    def make_basic_plot(self, x, y, savefig='plot.pdf',\
        x_error=[], plot_kwargs={}, fig_kwargs={}):

        """
        Make a basic plot.

        Inputs:
        x [list or array]: x-axis data to be plotted
        y [list or array]: y-axis data to be plotted
        
        Optional:
        x_error [list or array]: x error bars
        savefig [str]: Name of saved figure
        plot_kwargs [dict]: pass any kwargs to plt.plot()
        fig_kwargs [dict]: pass any kwargs to plt.gca().set()
        """
        
        # Setup figure:
        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()

        # Main plot:
        plt.loglog(x, y, **plot_kwargs)
            
        # Include x error bars:
        if len(x_error) != 0:
            
            # Remove label if defined:
            if "label" in plot_kwargs.keys():
                plot_kwargs["label"] = "_nolabel_"

            plt.errorbar(x, y, xerr=x_error, **plot_kwargs)

        # axes and labels:
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color="grey",alpha=0.3,ls="--")
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.set(**fig_kwargs)
        if "label" in plot_kwargs.keys():
            plt.legend(loc=1,frameon=True)
        
        # Save and show:
        plt.savefig(savefig)
        plt.show()
        plt.close()

        return
