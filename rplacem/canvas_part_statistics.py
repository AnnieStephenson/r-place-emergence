import numpy as np
import rplacem.variables_rplace2022 as var
import rplacem.compute_variables as comp

class CanvasPartStatistics(object):
    ''' Object containing all the important time-dependent variables concerning the input CanvasPart
    
    attributes
    ----------
    id : str
        Corresponds to the out_name() of the CanvasPart. 
        Is unique for compositions (where it is = cpart.id) and for rectangles (cpart.is_rectangle == True)
        Can designate different canvas parts in other cases.
    n_t_bins: int
        number of time intervals
    t_interval: float
        width of time interval
    t_ranges:
        1d array of size n_t_bins+1
    tmin, tmax:
        time range on which to compute the variables
    mean_stability:
        stability averaged over time and over all pixels of the CanvasPart.
    stability: 1d numpy array
        stability averaged over all pixels of the CanvasPart, for all time intervals.
    '''

    def __init__(self,
                 cpart,
                 nbins=80,
                 tmin=0,
                 tmax=var.TIME_WHITEONLY, # rejecting period with only white pixels
                 vars_to_compute={'stability': True},
                 verbose=False
                 ):

        self.id = cpart.out_name() 

        self.tmin = tmin
        self.tmax = tmax
        self.n_t_bins = nbins
        self.t_interval = tmax / nbins # seconds
        self.t_ranges = np.arange(tmin, tmax + self.t_interval - 1e-4, self.t_interval) 

        if vars_to_compute['stability']:
            if verbose:
                print('computing stability')
            # mean stability
            self.mean_stability = comp.stability(cpart, np.asarray([tmin, tmax]), False,False,False,True)[0][0]

            # time dependent stability
            self.stability = comp.stability(cpart, self.t_ranges, False,False,False,True)[0]
        
        