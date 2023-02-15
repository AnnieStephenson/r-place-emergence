import numpy as np
import os
import rplacem.variables_rplace2022 as var
import rplacem.compute_variables as comp
import rplacem.utilities as util

class CanvasPartStatistics(object):
    ''' Object containing all the important time-dependent variables concerning the input CanvasPart
    
    attributes
    ----------
    id : str
        Corresponds to the out_name() of the CanvasPart. 
        Is unique for compositions (where it is = cpart.id) and for rectangles (cpart.is_rectangle == True)
        Can designate different canvas parts in other cases.
    area : int
        number of pixels of the canvas part
    n_t_bins: int
        number of time intervals
    t_interval: float
        width of time interval
    t_ranges:
        1d array of size n_t_bins+1
    tmin, tmax:
        time range on which to compute the variables
    stability: float
        stability averaged over time and over all pixels of the CanvasPart.
    stability_vst: 1d numpy array
        stability averaged over all pixels of the CanvasPart, for all time intervals.
    '''

    def __init__(self,
                 cpart,
                 n_tbins=80,
                 tmin=0,
                 tmax=var.TIME_WHITEONLY, # rejecting period with only white pixels
                 compute_vars={'stability': 4, 'mean_stability': 4, 'entropy' : 4},
                 verbose=False,
                 renew=True
                 ):
        '''
        compute_vars says what level of information to compute and store for each variable.
            0: not computed
            1: basic variable is computed
            2: images are created and stored in the form of 2d numpy arrays of pixels containing color indices
            3: these pixel arrays are transformed to images and stored
            4: more extensive info is stored in the class and in a pickle file
        verbose: boolean saying if output is printed during running
        renew: delete the output directories if they exist, before re-running
        '''

        self.id = cpart.out_name() 
        if compute_vars['stability'] < 3 and compute_vars['mean_stability'] < 3 and compute_vars['entropy'] < 3:
            renew = False
        if renew:
            util.make_dir(os.path.join(var.FIGS_PATH, cpart.out_name()), renew=True)
        self.compute_vars = compute_vars
        self.verbose = verbose
        self.area = len(cpart.pixel_changes)

        self.tmin = tmin
        self.tmax = tmax
        self.n_t_bins = n_tbins
        self.t_interval = tmax / n_tbins # seconds
        self.t_ranges = np.arange(tmin, tmax + self.t_interval - 1e-4, self.t_interval) 

        # mean stability
        self.compute_mean_stability(cpart, compute_vars['mean_stability'])
        # stability vs time
        self.compute_stability_vst(cpart, compute_vars['stability'])
        # entropy and images versus time
        self.compute_entropy(cpart, compute_vars['entropy'])


    def compute_mean_stability(self, cpart, level):
        if level == 0:
            return
        if self.verbose:
            print('computing mean stability')

        res = comp.stability(cpart, np.asarray([self.tmin, self.tmax]), level>1, level>2, level>3, True, self.verbose)
        self.stability = res[0][0]
        self.stable_image = res[1][0] if level > 1 else None
        self.second_stable_image = res[2][0] if level > 1 else None
        self.third_stable_image = res[3][0] if level > 1 else None

    def compute_stability_vst(self, cpart, level):
        if level == 0:
            return
        if self.verbose:
            print('computing stability vs time')

        # time dependent stability
        res = comp.stability(cpart, self.t_ranges, level>1, level>2, level>3, True, self.verbose)
        self.stability_vst = res[0]
        self.stable_image_vst = res[1] if level > 1 else None
        self.second_stable_image_vst = res[2] if level > 1 else None
        self.third_stable_image_vst = res[3] if level > 1 else None

    def compute_entropy(self, cpart, level):
        if level == 0:
            return
        if self.verbose:
            print('computing entropy and images versus time')

        # time dependent stability
        res = comp.save_part_over_time(cpart, self.t_ranges, level>1, True, level<3, False, self.verbose)
        size_bmp = res[0]
        size_png = res[1]
        self.true_image_vst = res[2] if level > 1 else None
        self.entropy_vst = size_png / size_bmp
        print ('size_bmp / # pixels = ', size_bmp / self.area)

        