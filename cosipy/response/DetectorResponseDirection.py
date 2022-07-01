from histpy import Histogram, Axes, Axis

class DetectorResponseDirection:

    def __init__(self, axes, contents):

        self._hist = Histogram(axes, contents = contents)
        
    
        
