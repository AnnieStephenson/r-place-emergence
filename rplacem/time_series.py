import numpy as np
import os

class TimeSeries(object):
    '''
    Object recording values for a time-dependent variable.

    attributes
    ----------

    methods
    -------
    private:
        __init__
    protected:
    public:
        t_ranges()
    '''

    def __init__(self,
                 val=None,
                 cpstats=None,
                 t_interval=300,
                 tmin=0,
                 sw_width=40):
        
        self.val = val
        if val == None:
            self.exists = False
        else:
            self.exists = True
            self.n_pts = len(val)
            if cpstats == None:
                self.tmin = tmin
                self.t_interval = t_interval
                self.sw_width = sw_width
            else:
                self.tmin = cpstats.tmin
                self.t_interval = cpstats.t_interval
                self.sw_width = cpstats.sw_width

    def ratio_to_sw_average(self):
        '''
        At time index i, returns ratio of self.val[i] to the average over the preceding sliding window [i-sw_width : i[.
        The average from 0 to i-1 is used when i < sw_width.
        The cumulative_sum method is much faster than other methods.
        '''
        mean_sliding = np.empty(self.n_pts)
        sw = self.sw_width
        cumul_sum = np.cumsum(self.val) # cumsum[i] is the sum of values in indices [0, i] with i included
        mean_sliding[0] = self.val[0]
        mean_sliding[1:(sw+1)] = cumul_sum[0:sw] / np.arange(1, sw+1)
        mean_sliding[(sw+1):] = (cumul_sum[sw:] - cumul_sum[:-sw]) / float(sw) #CHECK THAT BEG IDX

        return self.val / mean_sliding

    def t_ranges(self):
        return np.arange(self.tmin, self.tmin + self.n_pts * self.t_interval - 1e-4, self.t_interval)