import numpy as np
import os
import pandas as pd
import rplacem.variables_rplace2022 as var
import rplacem.plot_utilities as plot


class TimeSeries(object):
    '''
    Object recording values for a time-dependent variable.

    attributes
    ----------
    n_pts: int
        number of limits of time bins (ie n_timebins+1)
    val: numpy 1d array, length n_pts
        values of the variables at each timestep
    t_ranges: numpy 1d array, length n_pts
        time limits at which the values of val are given
        This is filled only if set_t_ranges() is run (if record_all==True in __init__())
    tmin: float
        minimum time (first element of t_ranges)
    t_interval: float
        time interval (in seconds) between two times of t_ranges
    sw_width_mean: int
        width of the sliding window used for the ratio-to-sliding average, as a number of time intervals
    sw_width_ews: int
        width of the sliding window used for calculating variance/autocorrelation/skewness, as a number of time intervals
    desc_long: string
        long description of the meaning of the variable
    desc_short: string
        short description of the variable, used as y-axis label for plotting
    name: string
        generic name of the variable
    savename: string (without spaces)
        core of the name under which the plot is saved (does not include directory and file extension)
    ratio_to_sw_mean:
        At index i, contains the ratio of self.val[i] to the average over the
        preceding sliding window [max(0, i-sw_width_mean) : i[  (excluding i)
        Set in set_ratio_to_sw_average()
    variance,
    skewness,
    autocorrelation: numpy 1d array, length n_pts
        At index i, it contains the relevant transformation applied to the
        values at indices [max(0, i-sw_width_ews) : i]  (including i)
        Set in set_variance(), set_autocorrelation(), set_skewness()

    methods
    -------
    private:
        __init__
    protected:
    public:
        exists(): returns bool
            True if val is filled (not None)
        set_all_vars()
        set_t_ranges()
        set_ratio_to_sw_average()
        set_variance()
        set_autocorrelation()
        set_skewness()
    '''

    def __init__(self,
                 val=None,
                 cpstat=None,
                 t_interval=300,
                 tmin=0,
                 sw_width_mean=40,
                 sw_width_ews=10,
                 desc_long='',
                 desc_short='',
                 name='',
                 savename='',
                 record_all=False
                 ):

        self.val = val
        if self.exists():
            self.n_pts = len(val)
            if cpstat is None:
                self.tmin = tmin
                self.t_interval = t_interval
                self.sw_width_mean = sw_width_mean
            else:
                self.tmin = cpstat.tmin
                self.t_interval = cpstat.t_interval
                self.sw_width_mean = cpstat.sw_width
            self.sw_width_ews = sw_width_ews

        self.desc_long = desc_long
        self.desc_short = desc_short
        self.name = name
        self.savename = os.path.join(var.FIGS_PATH, cpstat.id, savename + '.png') if savename != '' else ''

        self.t_ranges = None
        if record_all:
            self.set_all_vars()

    def exists(self):
        return np.any(self.val is not None)

    def set_all_vars(self):
        self.set_t_ranges()
        self.set_ratio_to_sw_average()
        self.set_variance()
        self.set_autocorrelation()
        self.set_skewness()

    def set_ratio_to_sw_average(self):
        '''
        At time index i, returns ratio of self.val[i] to the average over the preceding sliding window [i-sw_width_mean : i[.
        The average from 0 to i-1 is used when i < sw_width_mean.
        The cumulative_sum method is much faster than other methods.
        '''
        mean_sliding = np.empty(self.n_pts)
        sw = self.sw_width_mean
        cumul_sum = np.cumsum(self.val)  # cumsum[i] is the sum of values in indices [0, i] with i included
        mean_sliding[0] = self.val[0]
        mean_sliding[1:(sw+1)] = cumul_sum[0:sw] / np.arange(1, sw+1)
        mean_sliding[(sw+1):] = (cumul_sum[sw:] - cumul_sum[:-sw]) / float(sw)  # CHECK THAT BEG IDX

        self.ratio_to_sw_mean = self.val / mean_sliding

    def set_t_ranges(self):
        self.t_ranges = np.arange(self.tmin, self.tmin + self.n_pts * self.t_interval - 1e-4, self.t_interval)

    def set_variance(self):
        '''
        calculates the variance vs time of the state variable
        '''
        x = pd.Series(self.val)
        variance = x.rolling(window=self.sw_width_ews, min_periods=1).var()
        self.variance = np.array(variance)

    def set_skewness(self):
        '''
        calculates the skewness vs time of the state variable
        '''
        x = pd.Series(x)
        skewness = x.rolling(window=self.sw_width_ews, min_periods=1).skew()
        self.skewness = np.array(skewness)

    def set_autocorrelation(self):
        '''
        calculates the autocorrelation vs time of the state variable
        '''
        x = pd.Series(self.val)
        autocorrelation = x.rolling(window=self.sw_width_ews, min_periods=1).apply(lambda y: y.autocorr())
        self.autocorrelation = np.array(autocorrelation)

    def plot1d(self, xlog=False, xmin=None, ylog=False, ymin=None, ymax=None, save=True, hline=None, vline=None, ibeg_remove=0, iend_remove=0):
        if self.t_ranges is None:
            self.set_t_ranges()

        iend = self.n_pts - iend_remove
        plot.draw_1d(self.t_ranges[ibeg_remove:iend], self.val[ibeg_remove:iend],
                     xlab='Time [s]', ylab=self.desc_short,
                     xlog=xlog, xmin=xmin,
                     ylog=ylog, ymin=ymin, ymax=ymax,
                     hline=hline, vline=vline,
                     save=(self.savename if save else '')
                     )
