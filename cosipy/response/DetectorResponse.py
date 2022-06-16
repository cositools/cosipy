import argparse
import textwrap

class DetectorResponse:
    """
    DetectorResponse handles the multi-dimensional matrix that describes the
    full response of the instruments.

    Parameters
    ----------
    filename : str, Path
        Path to RSP file
    """
    
    def __init__(self, filename = None):
        self._filename = filename
    
    def dump(self):
        """
        Print the content of the response to stdout.
        """

        print(f"Filename: {self._filename}. No contents for now!")

def cosi_rsp_dump(argv = None):
    """
    Print the content of a detector response to stdout.
    """
    
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
    
    dr =  DetectorResponse(args.filename)
  
    dr.dump()
        
        
        
