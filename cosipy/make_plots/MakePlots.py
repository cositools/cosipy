# Imports:
import matplotlib.pyplot as plt

class MakePlots:

    def make_basic_plot(self, x, y, xlabel='x', ylabel='y', legend='legend', \
            savefig='plot.pdf', x_error=[], marker="o", ls=""):

        """
        Make a basic plot.

        Inputs:
        x: x-axis data to be plotted (list or array)
        y: y-axis data to be plotted (list or array)
        
        Optional:
        x_error: x error bars (list or array)
        xlabel: Label for x-axis 
        ylabel: Label for y-axis
        legend: Label for legend
        savefig: Name of saved figure
        marker: plot marker style
        ls: plot line style
        """
        
        # Setup figure:
        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()

        # Main plot:
        plt.loglog(x, y, \
            marker=marker, ls=ls, color='black', label=legend)
            
        # Include x error bars:
        if len(x_error) != 0:
            plt.errorbar(x, y, xerr=x_error,\
                marker=marker, ls=ls, color='black', label="_nolabel_")

        # axes and labels:
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(color="grey",alpha=0.3,ls="--")
        plt.legend(loc=1,frameon=True)
        
        # Save and show:
        plt.savefig(savefig)
        plt.show()
        plt.close()

        return
