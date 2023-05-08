import numpy as np
import os
import cv2
import rplacem.variables_rplace2022 as var
import rplacem.compute_variables as comp
import rplacem.utilities as util
import rplacem.transitions as tran
import shutil
import warnings
import math


class CanvasPartStatistics(object):
    ''' Object containing all the important time-dependent variables concerning the input CanvasPart

    attributes
    ----------
    GENERAL
    id : str
        Corresponds to the out_name() of the CanvasPart.
        Is unique for compositions (where it is = cpart.id) and for rectangles (cpart.is_rectangle == True)
        Can designate different canvas parts in other cases.
            Set in __init__()
    compute_vars: dictionary (string to int)
        says what level of information to compute and store for each variable.
        The keys for the computed variables are:
            'stability', 'mean_stability', 'entropy', 'transitions', 'attackdefense'
        The levels mean:
            0: not computed
            1: basic variables are computed
            2: images are created and stored in the form of 2d numpy arrays of pixels containing color indices.
                Variables based on these images are also computed
            3: these pixel arrays are transformed to images and stored. Also makes a movie from stored images.
            4: more extensive info is stored in the class and in a pickle file.
        Set in __init__()
    verbose : boolean
        Saying if output is printed during running member functions
            Set in __init__()

    AREA
    area : int
        number of pixels (any pixels that are active at some time point) of the canvas part
            Set in __init__()
    area_rectangle : int
        size (in pixels) of the smallest rectangle encompassing the whole composition.
        Equals area when the composition is such a rectangle at some time point.
            Set in __init__()
    area_vst : 1d array of ints of length n_t_bins
        Number of active pixels at each time step.
            Needs compute_vars['attackdefense'] > 0
            Set in comp.n_changes_and_users(), in count_attack_defense_events().

    TIME BINS
    n_t_bins: int
        number of time intervals
            Set in __init__()
    t_interval: float
        width of time interval
            Set in __init__()
    t_ranges : 1d array of size n_t_bins+1
        limits of the time bins in which all variables are computed
           Set in __init__()
    t_unit : float
        time unit used for normalizing some variables.
            Set in __init__()
    t_norm : float
        time normalisation (t_interval / t_unit).
            Set in __init__()
    tmin, tmax: floats
        time range on which to compute the variables. As for now, cannot accept tmin > 0.
            Set in __init__()
    sw_width: float
        Size [in seconds] of the sliding window over which the sliding reference image is computed

    TRANSITIONS
    transition_param : list of floats
        cutoff parameters for the tran.find_transitions() function;
        The last two arguments are in seconds

    VARIABLES
    STABILITY
    stability: 1d numpy array of floats
        stability averaged over all pixels of the CanvasPart, for each time bin.
            Needs compute_vars['stability'] > 0
            Set in comp.main_variables()
    instability_norm : 1d numpy array of floats
        Time-normalised instability: (1 - stability_vst) / t_norm
            Needs compute_vars['stability'] > 0
            Set in __init__()
    stable_image
    second_stable_image
    third_stable_image : 1d array of pixels images (2d numpy array containing color indices)
        Most stable image (and second and third most stable images) in each time bin
            Needs compute_vars['stability'] > 1
            Set in comp.main_variables()

    ENTROPY -- all need compute_vars['entropy'] > 0
    diff_pixels_stable_vs_ref
    diff_pixels_inst_vs_ref
    diff_pixels_inst_vs_inst
    diff_pixels_inst_vs_stable : 1d numpy array of length n_t_bins+1
        Number of pixels that differ between images:
        stable over the timestep, instantaneous at the end of this or the previous timestep ("inst"),
        or the reference image over the sliding window (ref)
            Set in comp.main_variables()
    frac_pixdiff_stable_vs_ref
    frac_pixdiff_inst_vs_ref
    frac_pixdiff_inst_vs_inst_norm
    frac_pixdiff_inst_vs_stable_norm : 1d numpy array of length n_t_bins
        Same as above, but divided the number of active pixels.
        Also normalized by t_norm for the last two.
            Set in ratios_and_normalizations()
    size_bmp
    size_png : 1d numpy array of size n_t_bins+1
        Size of the bmp or png image (of the canvas part) file at the end of each time step
            Needs compute_vars['entropy'] > 0 (or > 3 for size_png)
            Set in comp.main_variables()
    entropy_bpmnorm : 1d numpy array of size n_t_bins+1
        Ratio of sizes of the png and bmp image files at the end of each time step
            Set in ratios_and_normalizations()
    entropy : 1d numpy array of size n_t_bins
        Ratio of size of the png image file to the number of active pixels, at each time step
            Set in ratios_and_normalizations()
    true_image: 1d array of size n_t_bins, of 2d numpy arrays.
        Each element is a pixels image info (2d numpy array containing color indices)
        True image of the canvas part at the end of the time interval
            Needs compute_vars['entropy'] > 1
            Set in comp.main_variables()

    TRANSITIONS -- all need compute_vars['transitions'] > 0
    n_transitions : int
        Number of found transitions
            Set in search_transitions()
    transition_tinds
    transition_times: 2d array, shape (number of transitions, 6)
        Delimiting times, and indices of t_ranges, for each transition.
        For each transition, is of the form
        [beg, end of pre stable period, beg, end of transition period, beg, end of post stable region]
            Set in tran.find_transitions(), in search_transitions()
    trans_start_time
    trans_start_tind : array of size n_transitions
        Time (and indices in t_ranges) at which most of the variables characterizing the transition have reached
        their half maximum, as computed in tran.transition_start_time()
            Set in search_transitions()
    refimage_pretrans
    refimage_intrans
    refimage_posttrans : array of size min(1, number of transitions), of 2d numpy arrays.
        Each element is a pixels image info (2d numpy array containing color indices)
        Reference stable image over the sliding window before the transition (sliding window ending at the end
        of the pre-transition stable period) or after the transition (sliding window starting at the beginning
        of the post-transition stable period).
        refimage_intrans is the true_image at time trans_start_time
            Needs compute_vars['transitions'] > 1
            Set in search_transitions()
    frac_diff_pixels_pre_vs_post_trans : 1d array of shape (n_transitions)
        Fraction of the active pixels that differ between the pre- and post-transition stable images
            Needs compute_vars['transitions'] > 1
            Set in search_transitions()

    ATTACK-DEFENSE -- all need compute_vars['attackdefense'] > 0
    refimage_sw : 1d array of 2D pixels image (length n_t_bins+1)
        For each timestep of index i, it is the stable image over the interval [i - sw_width, i],
        used as reference image for all the attack/defense variables
        Kept if compute_vars['attackdefense'] > 1
            Set in comp.main_variables()
    n_changes : 1d array of length n_t_bins
        Number of pixel changes in each time bin
            Set in comp.main_variables()
    n_changes_norm :
        same as n_changes, normalized by t_norm and area_vst
            Set in ratios_and_normalizations()
    frac_attack_changes : 2d array of floats, shape (# transitions, n_t_bins)
        Fraction of pixel changes that are attacking (compared to sliding window reference), in each time bin
            Set in ratios_and_normalizations()
    n_users_total : float
        total number of user having contributed to the composition between tmin and tmax
            Set in comp.main_variables()
    n_users : 1d array of int of length n_t_bins
        number of users that changed any pixels in this composition in each time range
            Set in comp.main_variables()
    n_users_norm :
        same as n_users_vst, normalized by t_norm and area_vst
            Set in ratios_and_normalizations()
    frac_attackonly_users
    frac_defenseonly_users
    frac_bothattdef_users : 1d array of floats of length n_t_bins
        fraction of n_users that contributed pixels that are consistent refimage_sw,
        or not (attackonly), or fraction of users that did both (bothattdef)
            Set in ratios_and_normalizations()
    n_defense_users
    n_attack_users
    n_bothattdef_users :
        Same as above, but without divinging by n_users.
        Kept if compute_vars['attackdefense'] > 3
            Set in comp.main_variables()
    frac_attack_changes_image : array of size n_t_bins, of 2d numpy arrays.
        Each element is a pixels image info (2d numpy array containing color indices)
        Image containing, for each pixel, the fraction of pixel changes that are attacking, for each time step
        Needs compute_vars['attackdefense'] > 1
    returntime : 2d array of shape (n_t_bins+1, # pixels)
        For each time step, contains the time that each pixel spent in an 'attack' color
        during its latest or current attack (maxxed at the size of the sliding window)
        Kept if compute_vars['attackdefense'] > 3
            Set in comp.main_variables()
    returntime_tbinned : 2d array of shape (n_t_bins+1, n_returnt_bins)
        For each time bin, contains histogram of return times for all pixels
            Set in returntime_stats()
    returntime_mean
    returntime_median_overln2
    returntime_percentile90_overln2 : 2d array of shape (n_t_bins)
        mean, median, and 90th percentile of returntime in each time bin
            Set in returntime_stats()
    cumul_attack_timefrac : 1d array of shape n_t_bins+1
        Sum of the times that each pixel spent in an attack color during this timestep
        Normalized by the timestep width times #pixels
            Set in comp.main_variables()

    methods
    -------
    private:
        __init__
    protected:
    public:
        checks_warnings()
        ratios_and_normalizations()
            Normalize variables
        returntime_stats()
            Calculate various variables for the full returntime array
        search_transitions()
            Look for transitions and set associated attributes
    '''

    def __init__(self,
                 cpart,
                 n_tbins=750,
                 tmax=var.TIME_TOTAL,
                 compute_vars={'stability': 3, 'entropy': 3, 'transitions': 3, 'attackdefense': 3},
                 sliding_window=14400,
                 returnt_binwidth=100,
                 trans_param=[0.2, 0.05, 7200, 10800],
                 timeunit=300,  # 5 minutes
                 verbose=False,
                 renew=True,
                 dont_keep_dir=False
                 ):
        '''
        cpart : CanvasPart object
            That associated to this CanvasPartStatistics
        renew : boolean
            Delete the output directories if they exist, before re-running
        dont_keep_dir : boolean
            Remove directory after running, if it did not exist before
        compute_vars : string to int dictionary
            Gives level of information to compute for each variable. Check class doc for details.
        verbose: boolean
            Is output printed during running. Check class doc for details.
        trans_param : list of floats
            Fills transition_param attribute. Check class doc for details.
        tmax : float
            Fills tmax attribute. Check class doc for details.
        timeunit: float
            Fills t_unit attribute. Check class doc for details.
        n_tbins : int
            Fills n_t_bins attribute. Check class doc for details.
        sliding_window: float, in seconds
            Duration of the sliding window on which the reference image is computed
        returnt_binwidth : float, in seconds
            Size of the bins for the return time output histogram
        '''

        self.id = cpart.out_name()
        if compute_vars['stability'] < 3 and compute_vars['entropy'] < 3:
            renew = False
        dirpath = os.path.join(var.FIGS_PATH, str(cpart.out_name()))
        dir_exists = util.make_dir(dirpath, renew)

        self.compute_vars = compute_vars
        self.verbose = verbose
        self.area = len(cpart.coords[0])
        self.area_rectangle = cpart.width(0) * cpart.width(1)

        self.tmin = 0  # not functioning yet for tmin > 0
        self.tmax = tmax
        self.n_t_bins = n_tbins
        self.t_interval = (self.tmax - self.tmin) / n_tbins  # seconds
        self.t_unit = timeunit
        self.t_norm = self.t_interval / self.t_unit
        self.t_ranges = np.arange(self.tmin, tmax + self.t_interval - 1e-4, self.t_interval)
        self.sw_width = int(sliding_window/self.t_interval)

        # Creating attributes that will be computed in comp.main_variables()
        self.diff_pixels_stable_vs_ref = None
        self.diff_pixels_inst_vs_ref = None
        self.diff_pixels_inst_vs_inst = None
        self.diff_pixels_inst_vs_stable = None
        self.area_vst = None

        self.stability = None
        self.stable_image = None
        self.second_stable_image = None
        self.third_stable_image = None
        self.refimage_sw = None
        self.true_image = None
        self.frac_attack_changes_image = None

        self.n_changes = None
        self.n_defense_changes = None
        self.n_users = None
        self.n_users_total = None
        self.n_bothattdef_users = None
        self.n_defense_users = None
        self.returntime = None
        self.cumul_attack_timefrac = None

        self.size_png = None
        self.size_bmp = None

        self.transition_param = trans_param
        self.n_transitions = None

        self.checks_warnings()

        # Magic happens here
        comp.main_variables(cpart, self, print_progress=self.verbose, delete_dir=dont_keep_dir)

        # ratio variables and normalizations
        self.ratios_and_normalizations()

        # Return time: histogram, mean and median
        if self.compute_vars['attackdefense'] > 0:
            self.returntime_stats(sliding_window, binwidth=returnt_binwidth)

        # Make movies
        if self.compute_vars['stability'] > 2:
            util.save_movie(os.path.join(dirpath, 'VsTimeStab'), fps=15)
        if self.compute_vars['entropy'] > 2:
            util.save_movie(os.path.join(dirpath, 'VsTime'), fps=15)
        if self.compute_vars['attackdefense'] > 2:
            util.save_movie(os.path.join(dirpath, 'attack_defense_ratio'), fps=15)

        # Memory savings here
        if compute_vars['attackdefense'] < 4:
            self.returntime = None
            self.n_defense_changes = None
            self.n_defense_users = None
            self.n_bothattdef_users = None
        if compute_vars['entropy'] < 4:
            self.size_png = None
            self.diff_pixels_stable_vs_ref = None
            self.diff_pixels_inst_vs_ref = None
            self.diff_pixels_inst_vs_inst = None
            self.diff_pixels_inst_vs_stable = None
        if compute_vars['attackdefense'] < 2:
            self.refimage_sw = None

        # find transitions
        if compute_vars['transitions'] > 0:
            self.search_transitions(cpart, sliding_window=sliding_window)

        # remove directory if it did not exist before and if dont_keep_dir
        if (not dir_exists) and dont_keep_dir:
            shutil.rmtree(dirpath)

    def checks_warnings(self):
        if np.any(np.diff(self.t_ranges) < 0):
            warnings.warn('The array of time limits t_ranges should contain increasing values. It will be modified to fit this requirement.')
            for i in range(1, self.n_t_bins + 1): # need for loop because it needs to be done in that order
                if self.t_ranges[i] < self.t_ranges[i-1]:
                    self.t_ranges[i] = self.t_ranges[i-1]
        if self.sw_width == 0:
            warnings.warn('The interval size is larger than the sliding window. Choose a smaller interval size or a larger sliding window.')

    def ratios_and_normalizations(self):
        self.instability_norm = (1 - self.stability) / self.t_norm
        self.n_changes_norm = util.divide_treatzero(self.n_changes / self.t_norm, self.area_vst, 0, 0)
        self.n_users_norm = util.divide_treatzero(self.n_users / self.t_norm, self.area_vst, 0, 0)
        self.frac_pixdiff_stable_vs_ref = util.divide_treatzero(self.diff_pixels_stable_vs_ref, self.area_vst, 0, 0)
        self.frac_pixdiff_inst_vs_ref = util.divide_treatzero(self.diff_pixels_inst_vs_ref, self.area_vst, 0, 0)
        self.frac_pixdiff_inst_vs_inst_norm = util.divide_treatzero(self.diff_pixels_inst_vs_inst / self.t_norm, self.area_vst, 0, 0)
        self.frac_pixdiff_inst_vs_stable_norm = util.divide_treatzero(self.diff_pixels_inst_vs_stable / self.t_norm, self.area_vst, 0, 0)

        # attack-defense ratios
        self.frac_attack_changes = util.divide_treatzero(self.n_changes - self.n_defense_changes, self.n_changes, 0.5, 0.5)
        self.frac_defenseonly_users = util.divide_treatzero(self.n_defense_users - self.n_bothattdef_users, self.n_users, 0.5, 0.5)
        self.frac_bothattdef_users = util.divide_treatzero(self.n_bothattdef_users, self.n_users, 0.5, 0.5)
        self.frac_attackonly_users = util.divide_treatzero(self.n_users - self.n_defense_users - self.n_bothattdef_users, self.n_users, 0.5, 0.5)
        # for entropy
        self.entropy = util.divide_treatzero(self.size_png, self.area_vst)
        self.entropy_bmpnorm = self.size_png / self.size_bmp
        self.entropy[0] = 0
        self.entropy_bmpnorm[0] = 0
        idx_dividebyzero = np.where(self.area_vst == 0)
        self.entropy[idx_dividebyzero] = self.entropy_bmpnorm[idx_dividebyzero] * 3.2  # typical factor hard-coded here

    def returntime_stats(self, sliding_window, binwidth=100):
        returnt_bins = np.arange(0, sliding_window+binwidth, binwidth)
        self.returntime_tbinned = np.zeros((self.n_t_bins+1, math.ceil(sliding_window/binwidth)))
        self.returntime_mean = np.zeros(self.n_t_bins+1)
        self.returntime_median_overln2 = np.zeros(self.n_t_bins+1)
        self.returntime_percentile90_overln2 = np.zeros(self.n_t_bins+1)
        for t in range(1, self.n_t_bins+1):
            self.returntime_tbinned[t], _ = np.histogram(self.returntime[t], bins=returnt_bins)
            if np.count_nonzero(self.returntime[t] < 0) > 0:
                warnings.warn('There are negative return times, this is a problem!')
            self.returntime_mean[t] = np.mean(self.returntime[t])
            self.returntime_median_overln2[t] = np.median(self.returntime[t]) / np.log(2)
            self.returntime_percentile90_overln2[t] = np.percentile(self.returntime[t], 90) / np.log(2)

    def search_transitions(self, cpart, sliding_window):
        par = self.transition_param
        transitions = tran.find_transitions(self.t_ranges, self.frac_pixdiff_inst_vs_ref,
                                             cutoff=par[0], cutoff_stable=par[1], len_stableregion=par[2], distfromtrans_stableregion=par[3],
                                             sliding_win=sliding_window)
        self.transition_tinds = transitions[0]
        self.transition_times = transitions[1]
        self.n_transitions = len(transitions[1])
        trans = self.compute_vars['transitions']

        self.refimage_pretrans = cpart.white_image(3, images_number=self.n_transitions) if trans > 1 else None
        self.refimage_intrans = cpart.white_image(3, images_number=self.n_transitions) if trans > 1 else None
        self.refimage_posttrans = cpart.white_image(3, images_number=self.n_transitions) if trans > 1 else None
        self.trans_start_time = np.empty(self.n_transitions)
        self.trans_start_tind = np.empty(self.n_transitions, dtype=np.int64)
        self.frac_diff_pixels_pre_vs_post_trans = np.empty(self.n_transitions) if trans > 1 else None

        for j in range(0, self.n_transitions):
            end_pretrans_sw_ind = min(self.transition_tinds[j][1] + 1, self.n_t_bins+1)
            end_posttrans_sw_ind = min(self.transition_tinds[j][4] + self.sw_width + 1, self.n_t_bins+1)

            self.trans_start_time[j] = tran.transition_start_time(self, j)[1] # take the mean here, but could be the median (among variables)
            self.trans_start_tind[j] = np.argmax(self.t_ranges >= self.trans_start_time[j])

            if trans > 1:
                self.frac_diff_pixels_pre_vs_post_trans[j] = comp.count_image_differences(self.refimage_pretrans[j], self.refimage_posttrans[j], cpart) / self.area
                self.refimage_intrans[j] = self.true_image[self.trans_start_tind[j]]
                self.refimage_pretrans[j] = self.refimage_sw[end_pretrans_sw_ind]
                self.refimage_posttrans[j] = self.refimage_sw[end_posttrans_sw_ind]

            if trans > 2:
                util.pixels_to_image(self.refimage_pretrans[j], cpart.out_name(), 'referenceimage_sw_pre_transition'+str(j) + '.png')
                util.pixels_to_image(self.refimage_intrans[j], cpart.out_name(), 'referenceimage_at_transition'+str(j) + '.png')
                util.pixels_to_image(self.refimage_posttrans[j], cpart.out_name(), 'referenceimage_sw_post_transition'+str(j) + '.png')
