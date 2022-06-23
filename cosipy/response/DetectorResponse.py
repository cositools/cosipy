import argparse
import textwrap

from histpy import Histogram

class DetectorResponse:
    """
    DetectorResponse handles the multi-dimensional matrix that describes the
    full response of the instruments.

    Parameters
    ----------
    filename : str, Path, optional
        Path to file
    """
    
    def __init__(self, filename = None):
        pass

    @classmethod
    def open(self, filename)
        """
        Open a detector response file.

        Parameters
        ----------
        filename : str, Path
            Path to HDF5 file
        """

        self._dr = Histogram.open(filename)

        #Verify
        if self._dr.axes.labels != ("Ei","NuLambda","Em","Phi","PsiChi","SigmaTau","Dist"):
            raise ValueError("Unknown histogram axes labels")

    def get_directional_response(def, coord, interp = True):
        """
        Get the 
        """
        pass
        
    def get_point_source_expectation(self, orientation):
        pass

    
    
    def dump(self):
        """
        Print the content of the response to stdout.
        """

        print(f"Filename: {self._filename}. No contents for now!")

def cosi_rsp_dump(argv = None):
    """
    Print the content of a detector response to stdout.
    """

    # Parse arguments from commandline
    aPar = argparse.ArgumentParser(
        usage = ("%(prog)s filename "
                 "[--help] [options]"),
        description = textwrap.dedent(
            """
            Dump DR contents
            """),
        formatter_class=argparse.RawTextHelpFormatter)

    aPar.add_argument('filename',
                      help="Path to instrument response")
    
    args = aPar.parse_args(argv)

    # Init and dump 
    dr =  DetectorResponse(args.filename)
  
    dr.dump()
        
        
        
