import argparse
import textwrap

class DetectorResponse:

    def __init__(self, filename):
        self._filename = filename
    
    def dump(self):

        print(f"Filename: {self._filename}. No contents for now!")

def cosi_rsp_dump(argv = None):

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
        
        
        
