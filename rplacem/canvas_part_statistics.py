import numpy as np
import os
import rplacem.variables_rplace2022 as var
import rplacem.compute_variables as comp
import rplacem.utilities as util
import rplacem.transitions as tran
import rplacem.time_series as ts
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
            'stability', 'other', 'entropy', 'transitions', 'attackdefense'
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
    area_vst : TimeSeries, n_pts = n_t_bins+1
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
    t_lims : 1d array of size n_t_bins+1
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
    sw_width_sec: float
        Size [in seconds] of the sliding window over which the sliding reference image is computed
    sw_width: int
        Size [in units of t_interval] of the sliding window over which the sliding reference image is computed
    returnt_binwidth: float
        bin width [in seconds] for the saved histogram of return times returntime_tbinned

    TRANSITIONS
    transition_param : list of floats
        cutoff parameters for the tran.find_transitions() function;
        The last two arguments are in seconds

    VARIABLES
    STABILITY
    stability: TimeSeries, n_pts = n_t_bins+1
        stability averaged over all pixels of the CanvasPart, for each time bin.
            Needs compute_vars['stability'] > 0
            Set in comp.main_variables()
    instability_norm : TimeSeries, n_pts = n_t_bins+1
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
    diff_pixels_stable_vs_swref
    diff_pixels_inst_vs_swref
    diff_pixels_inst_vs_swref_forwardlook
    diff_pixels_inst_vs_inst
    diff_pixels_inst_vs_stable : TimeSeries, n_pts = n_t_bins+1
        Number of pixels that differ between images:
        stable over the timestep, instantaneous at the end of this or the previous timestep ("inst"),
        or the reference image over the sliding window ("swref") 
        or over a forward-looking future sliding window ("swref_forwardlook")
            Set in comp.main_variables()
    frac_pixdiff_stable_vs_swref
    frac_pixdiff_inst_vs_swref
    frac_pixdiff_inst_vs_swref_forwardlook
    frac_pixdiff_inst_vs_inst_norm
    frac_pixdiff_inst_vs_stable_norm : TimeSeries, n_pts = n_t_bins+1
        Same as above, but divided the number of active pixels.
        Also normalized by t_norm for the last two.
            Set in ratios_and_normalizations()
    size_uncompressed :
    size_compressed : TimeSeries, n_pts = n_t_bins+1
        Size of the uncompressed or compressed image (of the canvas part) file at the end of each time step
            Needs compute_vars['entropy'] > 0 (or > 3 for size_compressed)
            Set in comp.main_variables()
    entropy_bpmnorm : TimeSeries, n_pts = n_t_bins+1
        Ratio of sizes of the png and bmp image files at the end of each time step
            Set in ratios_and_normalizations()
    entropy : TimeSeries, n_pts = n_t_bins+1
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
        Delimiting times, and indices of t_lims, for each transition.
        For each transition, is of the form
        [beg, end of pre stable period, beg, end of transition period, beg, end of post stable region]
            Set in tran.find_transitions(), in search_transitions()
    trans_start_time
    trans_start_tind : array of size n_transitions
        Time (and indices in t_lims) at which most of the variables characterizing the transition have reached
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
    n_changes : TimeSeries, n_pts = n_t_bins+1
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
    n_users : TimeSeries, n_pts = n_t_bins+1
        number of users that changed any pixels in this composition in each time range
            Set in comp.main_variables()
    n_users_norm :
        same as n_users_vst, normalized by t_norm and area_vst
            Set in ratios_and_normalizations()
    frac_attackonly_users
    frac_defenseonly_users
    frac_bothattdef_users : TimeSeries, n_pts = n_t_bins+1
        fraction of n_users that contributed pixels that are consistent refimage_sw,
        or not (attackonly), or fraction of users that did both (bothattdef)
            Set in ratios_and_normalizations()
    n_defense_users
    n_attack_users
    n_bothattdef_users :
        Same as above, but without divinging by n_users.
        Kept if compute_vars['attackdefense'] > 1
            Set in comp.main_variables()
    frac_attack_changes_image : array of size n_t_bins, of 2d numpy arrays.
        Each element is a pixels image info (2d numpy array containing color indices)
        Image containing, for each pixel, the fraction of pixel changes that are attacking, for each time step
        Needs compute_vars['attackdefense'] > 1
    returntime : 2d array of shape (n_t_bins+1, # pixels)
        For each time step, contains the time that each pixel spent in an 'attack' color
        during its latest or current attack (maxxed at the size of the sliding window)
        Kept if compute_vars['attackdefense'] > 1
            Set in comp.main_variables()
    returntime_tbinned : 2d array of shape (n_t_bins+1, n_returnt_bins)
        For each time bin, contains histogram of return times for all pixels
            Set in returntime_stats()
    returntime_mean
    returntime_median_overln2
    returntime_percentile90_overln2 : 2d array of shape (n_t_bins)
        mean, median, and 90th percentile of returntime in each time bin
            Set in returntime_stats()
    cumul_attack_timefrac : TimeSeries, n_pts = n_t_bins+1
        Sum of the times that each pixel spent in an attack color during this timestep
        Normalized by the timestep width times #pixels
            Set in comp.main_variables()

    OTHER: -- all need compute_vars['other'] > 0
    n_moderator_changes
    n_cooldowncheat_changes
    n_redundant_color_changes
    n_redundant_coloranduser_changes
        number of pixel changes that:
        - were issued by moderators
        - break the 5-minute per-user cooldown
        - are redundant in color with what the pixel contained before the change
        - are redundant in color and user with what the pixel contained before the change
            Set in comp.main_variables()
    frac_moderator_changes
    frac_cooldowncheat_changes
    frac_redundant_color_changes
    frac_redundant_coloranduser_changes
        Same as above, but divided by the number of pixel changes in this timestep
            Set in ratios_and_normalizations()

    methods
    -------
    private:
        __init__
    protected:
    public:
        ts_init()
        checks_warnings()
        keep_all_basics()
            Check if all basic variables are asked for (compute_vars[:] > 0)
        ratios_and_normalizations()
            Normalize variables
        returntime_stats()
            Calculate various variables for the full returntime array
        search_transitions()
            Look for transitions and set associated attributes
        fill_timeseries_info()
            User-called function that fills all the textual attributes of 1d time-series variables
    '''

    def __init__(self,
                 cpart,
                 t_interval=300,                 
                 tmax=var.TIME_TOTAL,
                 compute_vars={'stability': 3, 'entropy': 3, 'transitions': 3, 'attackdefense': 3, 'other': 1},
                 sliding_window=14400,
                 returnt_binwidth=100,
                 trans_param=[0.2, 0.05, 7200, 10800],
                 timeunit=300,  # 5 minutes
                 verbose=False,
                 renew=True,
                 dont_keep_dir=False,
                 flattening='hilbert_pkg',
                 compression='LZ77'
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
        t_interval : float, in seconds
            width of the time bins
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
        self.tmin = cpart.minimum_time()
        self.tmax = tmax
        self.t_interval = t_interval # seconds
        self.n_t_bins = math.ceil((self.tmax - self.tmin) / self.t_interval)
        self.t_lims = np.arange(self.tmin, tmax + self.t_interval, self.t_interval)
        self.t_unit = timeunit
        self.t_norm = self.t_interval / self.t_unit
        self.sw_width = int(sliding_window/self.t_interval)
        self.sw_width_sec = self.sw_width * self.t_interval # force the sliding window to be a multiple of the time interval
        self.returnt_binwidth = returnt_binwidth

        # Creating attributes that will be computed in comp.main_variables()
        self.diff_pixels_stable_vs_swref = ts.TimeSeries()
        self.diff_pixels_inst_vs_swref = ts.TimeSeries()
        self.diff_pixels_inst_vs_swref_forwardlook = ts.TimeSeries()
        self.diff_pixels_inst_vs_inst = ts.TimeSeries()
        self.diff_pixels_inst_vs_stable = ts.TimeSeries()
        self.area_vst = ts.TimeSeries()

        self.stability = ts.TimeSeries()
        self.stable_image = None
        self.second_stable_image = None
        self.third_stable_image = None
        self.refimage_sw = None
        self.true_image = None
        self.attack_defense_image = None
        self.frac_attack_changes_image = None

        self.n_changes = ts.TimeSeries()
        self.n_defense_changes = ts.TimeSeries()
        self.n_users = ts.TimeSeries()
        self.n_users_total = None
        self.n_bothattdef_users = ts.TimeSeries()
        self.n_defense_users = ts.TimeSeries()
        self.returntime = None
        self.cumul_attack_timefrac = ts.TimeSeries()

        self.n_moderator_changes = ts.TimeSeries()
        self.n_cooldowncheat_changes = ts.TimeSeries()
        self.n_redundant_color_changes = ts.TimeSeries()
        self.n_redundant_coloranduser_changes = ts.TimeSeries()
        
        self.size_compressed = ts.TimeSeries()
        self.size_uncompressed = ts.TimeSeries()

        self.transition_param = trans_param
        self.n_transitions = None

        self.checks_warnings()

        # Magic happens here
        comp.main_variables(cpart, self, print_progress=self.verbose, delete_dir=dont_keep_dir,
                            flattening=flattening,
                            compression=compression)

        # ratio variables and normalizations
        self.ratios_and_normalizations()

        # Return time: histogram, mean and median
        if self.compute_vars['attackdefense'] > 0:
            self.returntime_stats(self.sw_width_sec, binwidth=returnt_binwidth)

        # Make movies
        if self.compute_vars['stability'] > 2:
            util.save_movie(os.path.join(dirpath, 'VsTimeStab'), fps=15)
        if self.compute_vars['entropy'] > 2:
            util.save_movie(os.path.join(dirpath, 'VsTimeInst'), fps=15)
        if self.compute_vars['attackdefense'] > 2:
            util.save_movie(os.path.join(dirpath, 'attack_defense_ratio'), fps=15)
            util.save_movie(os.path.join(dirpath, 'attack_defense_Ising'), fps=6)

        # Memory savings here
        if compute_vars['attackdefense'] < 2:
            self.returntime = None
            self.n_defense_changes = ts.TimeSeries()
            self.n_defense_users = ts.TimeSeries()
            self.n_bothattdef_users = ts.TimeSeries()
        if compute_vars['entropy'] < 2:
            self.size_compressed = ts.TimeSeries()
            self.diff_pixels_stable_vs_swref = ts.TimeSeries()
            self.diff_pixels_inst_vs_swref_forwardlook = ts.TimeSeries()
            self.diff_pixels_inst_vs_inst = ts.TimeSeries()
            self.diff_pixels_inst_vs_stable = ts.TimeSeries()
        if compute_vars['attackdefense'] < 2:
            self.refimage_sw = None

        # find transitions
        if compute_vars['transitions'] > 0:
            self.search_transitions(cpart)

        # remove directory if it did not exist before and if dont_keep_dir
        if (not dir_exists) and dont_keep_dir:
            shutil.rmtree(dirpath)

    def checks_warnings(self):
        if np.any(np.diff(self.t_lims) < 0):
            warnings.warn('The array of time limits t_lims should contain increasing values. It will be modified to fit this requirement.')
            for i in range(1, self.n_t_bins + 1): # need for loop because it needs to be done in that order
                if self.t_lims[i] < self.t_lims[i-1]:
                    self.t_lims[i] = self.t_lims[i-1]
        if self.sw_width == 0:
            warnings.warn('The interval size is larger than the sliding window. Choose a smaller interval size or a larger sliding window.')

    def ts_init(self, val):
        return ts.TimeSeries(val=val, cpstat=self)
    
    def keep_all_basics(self):
        return self.compute_vars['entropy'] > 0 \
           and self.compute_vars['stability'] > 0 \
           and self.compute_vars['attackdefense']> 0 \
           and self.compute_vars['transitions'] > 0 \
           and self.compute_vars['other'] > 0
    
    def ratios_and_normalizations(self):
        self.instability_norm = self.ts_init( (1 - self.stability.val) / self.t_norm )
        self.n_changes_norm = self.ts_init( util.divide_treatzero(self.n_changes.val / self.t_norm, self.area_vst.val, 0, 0) )
        self.n_users_norm = self.ts_init( util.divide_treatzero(self.n_users.val / self.t_norm, self.area_vst.val, 0, 0) )
        self.frac_pixdiff_stable_vs_swref = self.ts_init( util.divide_treatzero(self.diff_pixels_stable_vs_swref.val, self.area_vst.val, 0, 0) )
        self.frac_pixdiff_inst_vs_swref = self.ts_init( util.divide_treatzero(self.diff_pixels_inst_vs_swref.val, self.area_vst.val, 0, 0) )
        self.frac_pixdiff_inst_vs_swref_forwardlook = self.ts_init( util.divide_treatzero(self.diff_pixels_inst_vs_swref_forwardlook.val, self.area_vst.val, 0, 0) )
        self.frac_pixdiff_inst_vs_inst_norm = self.ts_init( util.divide_treatzero(self.diff_pixels_inst_vs_inst.val / self.t_norm, self.area_vst.val, 0, 0) )
        self.frac_pixdiff_inst_vs_stable_norm = self.ts_init( util.divide_treatzero(self.diff_pixels_inst_vs_stable.val / self.t_norm, self.area_vst.val, 0, 0) )

        # attack-defense ratios
        self.frac_attack_changes = self.ts_init( util.divide_treatzero(self.n_changes.val - self.n_defense_changes.val, self.n_changes.val, 0.5, 0.5) )
        self.frac_defenseonly_users = self.ts_init( util.divide_treatzero(self.n_defense_users.val - self.n_bothattdef_users.val, self.n_users.val, 0.5, 0.5) )
        self.frac_bothattdef_users = self.ts_init( util.divide_treatzero(self.n_bothattdef_users.val, self.n_users.val, 0.5, 0.5) )
        self.frac_attackonly_users = self.ts_init( util.divide_treatzero(self.n_users.val - self.n_defense_users.val - self.n_bothattdef_users.val, self.n_users.val, 0.5, 0.5) )
        # for entropy
        self.entropy = self.ts_init( util.divide_treatzero(self.size_compressed.val, self.area_vst.val) )
        self.entropy_bmpnorm = self.ts_init( util.divide_treatzero(self.size_compressed.val, self.size_uncompressed.val, 0, 0) )
        self.entropy.val[0] = 0
        self.entropy_bmpnorm.val[0] = 0
        idx_dividebyzero = np.where(self.area_vst.val == 0)
        self.entropy.val[idx_dividebyzero] = self.entropy_bmpnorm.val[idx_dividebyzero] * 3.2  # typical factor hard-coded here
        # other
        self.frac_moderator_changes = self.ts_init( util.divide_treatzero(self.n_moderator_changes.val, self.n_changes.val, 0, 0) )
        self.frac_cooldowncheat_changes = self.ts_init( util.divide_treatzero(self.n_cooldowncheat_changes.val, self.n_changes.val, 0, 0) )
        self.frac_redundant_color_changes = self.ts_init( util.divide_treatzero(self.n_redundant_color_changes.val, self.n_changes.val, 0, 0) )
        self.frac_redundant_coloranduser_changes = self.ts_init( util.divide_treatzero(self.n_redundant_coloranduser_changes.val, self.n_changes.val, 0, 0) )

    def returntime_stats(self, sliding_window, binwidth=100):
        returnt_bins = np.arange(0, sliding_window+binwidth-1e-4, binwidth)
        self.returntime_tbinned = np.zeros((self.n_t_bins+1, math.ceil(sliding_window/binwidth)))
        self.returntime_mean = self.ts_init( np.zeros(self.n_t_bins+1) )
        self.returntime_median_overln2 = self.ts_init( np.zeros(self.n_t_bins+1) )
        self.returntime_percentile90_overln2 = self.ts_init( np.zeros(self.n_t_bins+1) )
        for t in range(1, self.n_t_bins+1):
            self.returntime_tbinned[t], _ = np.histogram(self.returntime[t], bins=returnt_bins)
            if np.count_nonzero(self.returntime[t] < 0) > 0:
                print(t, self.returntime[t], np.count_nonzero(self.returntime[t] < 0))
                warnings.warn('There are negative return times, this is a problem!')
            self.returntime_mean.val[t] = np.mean(self.returntime[t])
            self.returntime_median_overln2.val[t] = np.median(self.returntime[t]) / np.log(2)
            self.returntime_percentile90_overln2.val[t] = np.percentile(self.returntime[t], 90) / np.log(2)

    def search_transitions(self, cpart):
        par = self.transition_param
        transitions = tran.find_transitions(self.t_lims, self.frac_pixdiff_inst_vs_swref.val, self.frac_pixdiff_inst_vs_swref_forwardlook.val,
                                             cutoff=par[0], cutoff_stable=par[1], len_stableregion=par[2], distfromtrans_stableregion=par[3])
        self.transition_tinds = transitions[0]
        self.transition_times = transitions[1]
        self.n_transitions = len(transitions[1])
        trans = self.compute_vars['transitions']

        if trans > 1:
            self.refimage_pretrans = cpart.white_image(3, images_number=self.n_transitions) if trans > 1 else None
            self.refimage_intrans = cpart.white_image(3, images_number=self.n_transitions) if trans > 1 else None
            self.refimage_posttrans = cpart.white_image(3, images_number=self.n_transitions) if trans > 1 else None
            self.frac_diff_pixels_pre_vs_post_trans = np.empty(self.n_transitions) if trans > 1 else None
        self.trans_start_time = np.empty(self.n_transitions) if self.keep_all_basics() else None
        self.trans_start_tind = np.empty(self.n_transitions, dtype=np.int64) if self.keep_all_basics() else None

        for j in range(0, self.n_transitions):
            if self.keep_all_basics():
                self.trans_start_time[j] = tran.transition_start_time(self, j)[1] # take the mean here, but could be the median (among variables)
                self.trans_start_tind[j] = np.argmax(self.t_lims >= self.trans_start_time[j])

            if trans > 1:
                self.frac_diff_pixels_pre_vs_post_trans[j] = comp.count_image_differences(self.refimage_pretrans[j], self.refimage_posttrans[j], cpart) / self.area
                
                end_pretrans_sw_ind = min(self.transition_tinds[j][1] + 1, self.n_t_bins+1)
                end_posttrans_sw_ind = min(self.transition_tinds[j][4] + self.sw_width + 1, self.n_t_bins+1)
                self.refimage_intrans[j] = self.true_image[self.trans_start_tind[j]]
                self.refimage_pretrans[j] = self.refimage_sw[end_pretrans_sw_ind]
                self.refimage_posttrans[j] = self.refimage_sw[end_posttrans_sw_ind]

            if trans > 2:
                util.pixels_to_image(self.refimage_pretrans[j], cpart.out_name(), 'referenceimage_sw_pre_transition'+str(j) + '.png')
                util.pixels_to_image(self.refimage_intrans[j], cpart.out_name(), 'referenceimage_at_transition'+str(j) + '.png')
                util.pixels_to_image(self.refimage_posttrans[j], cpart.out_name(), 'referenceimage_sw_post_transition'+str(j) + '.png')

    def fill_timeseries_info(self):
        def filepath(name):
            return os.path.join(var.FIGS_PATH, self.id, name+'.png')
        n_min_tunit = str(int(self.t_unit+1e-3)) + ' min'

        self.frac_pixdiff_inst_vs_stable_norm.desc_long = 'Fraction of pixels differing from the stable image in the previous step, normalized by the time unit.'
        self.frac_pixdiff_inst_vs_stable_norm.desc_short = 'frac of pixels differing from previous-step stable image / '+n_min_tunit
        self.frac_pixdiff_inst_vs_stable_norm.label = 'frac_pixdiff_inst_vs_stable_norm'
        self.frac_pixdiff_inst_vs_stable_norm.savename = filepath('fraction_of_differing_pixels_vs_stable_normalized')

        self.frac_pixdiff_inst_vs_inst_norm.desc_long = 'Fraction of pixels differing from the instantaneous image in the previous step, normalized by the time unit.'
        self.frac_pixdiff_inst_vs_inst_norm.desc_short = 'frac of pixels differing from previous-step image / '+n_min_tunit
        self.frac_pixdiff_inst_vs_inst_norm.label = 'frac_pixdiff_inst_vs_inst_norm'
        self.frac_pixdiff_inst_vs_inst_norm.savename = filepath('fraction_of_differing_pixels_normalized')

        self.frac_pixdiff_inst_vs_swref.desc_long = 'Fraction of pixels differing from the reference image in the preceding sliding window'
        self.frac_pixdiff_inst_vs_swref.desc_short = 'frac of pixels differing from sliding-window ref image'
        self.frac_pixdiff_inst_vs_swref.label = 'frac_pixdiff_inst_vs_swref'
        self.frac_pixdiff_inst_vs_swref.savename = filepath('fraction_of_differing_pixels_vs_slidingwindowref')

        self.frac_pixdiff_inst_vs_swref_forwardlook.desc_long = 'Fraction of pixels differing from the reference image in the future (forward-looking) sliding window'
        self.frac_pixdiff_inst_vs_swref_forwardlook.desc_short = 'frac of pixels differing from forward-looking sliding-window ref image'
        self.frac_pixdiff_inst_vs_swref_forwardlook.label = 'frac_pixdiff_inst_vs_swref_forwardlook'
        self.frac_pixdiff_inst_vs_swref_forwardlook.savename = filepath('fraction_of_differing_pixels_vs_slidingwindowref_forwardlook')

        self.instability_norm.desc_long = 'instability = (1 - stability), normalized by the time unit. \
            Stability is the time fraction that each pixel spent in its dominant color, averaged over all active pixels'
        self.instability_norm.desc_short = 'instability / '+n_min_tunit
        self.instability_norm.label = 'instability_norm'
        self.instability_norm.savename = filepath('instability_normalized')

        self.n_users_norm.desc_long = 'number of users, normalized by the time unit and the number of active pixels.'
        self.n_users_norm.desc_short = '# users / area / '+n_min_tunit
        self.n_users_norm.label = 'n_users_norm'
        self.n_users_norm.savename = filepath('number_of_users_normalized')

        self.n_changes_norm.desc_long = 'number of pixel changes, normalized by the time unit and the number of active pixels.'
        self.n_changes_norm.desc_short = '# pixel changes / area / '+n_min_tunit
        self.n_changes_norm.label = 'n_changes_norm'
        self.n_changes_norm.savename = filepath('number_of_pixel_changes_normalized')

        self.entropy.desc_long = 'entropy, calculated as the computable information density of the instantaneous image'
        self.entropy.desc_short = 'entropy (computable information density)'
        self.entropy.label = 'entropy'
        self.entropy.savename = filepath('entropy')

        self.frac_attack_changes.desc_long = 'Fraction of the number of pixel changes that are attacking. \
            Attack is defined compared to the stable reference image on a sliding window.'
        self.frac_attack_changes.desc_short = 'fraction of attack changes'
        self.frac_attack_changes.label = 'frac_attack_changes'
        self.frac_attack_changes.savename = filepath('fraction_attack_pixelchanges')

        self.frac_attackonly_users.desc_long = 'Fraction of the active users that are only attacking in this timestep. \
            Attack is defined compared to the stable reference image on a sliding window.'
        self.frac_attackonly_users.desc_short = 'fraction of users only attacking'
        self.frac_attackonly_users.label = 'frac_attackonly_users'
        self.frac_attackonly_users.savename = filepath('fraction_of_users_onlyattacking')

        self.frac_defenseonly_users.desc_long = 'Fraction of the active users that are only defending in this timestep. \
            Defense is defined as following the stable reference image on a sliding window.'
        self.frac_defenseonly_users.desc_short = 'fraction of users only defending'
        self.frac_defenseonly_users.label = 'frac_defenseonly_users'
        self.frac_defenseonly_users.savename = filepath('fraction_of_users_onlydefending')

        self.frac_bothattdef_users.desc_long = 'Fraction of the active users that are both attacking and defending in this timestep. \
            Attack is defined compared to the stable reference image on a sliding window.'
        self.frac_bothattdef_users.desc_short = 'fraction of users both attacking and defending'
        self.frac_bothattdef_users.label = 'frac_bothattdef_users'
        self.frac_bothattdef_users.savename = filepath('fraction_of_users_bothattackingdefending')

        self.returntime_median_overln2.desc_long = 'Median time for pixels to recover from attack [s] / ln(2)'
        self.returntime_median_overln2.desc_short = 'median pixel recovery time from attack [s] / ln(2)'
        self.returntime_median_overln2.label = 'returntime_median_overln2'
        self.returntime_median_overln2.savename = filepath('median_pixel_recovery_time')

        self.returntime_mean.desc_long = 'Mean time for pixels to recover from attack [s]'
        self.returntime_mean.desc_short = 'mean pixel recovery time from attack [s]'
        self.returntime_mean.label = 'returntime_mean'
        self.returntime_mean.savename = filepath('mean_pixel_recovery_time')

        self.cumul_attack_timefrac.desc_long = 'Fraction of the time that all pixels spent in an attack color [s]'
        self.cumul_attack_timefrac.desc_short = 'frac of time spent in attack colors [s]'
        self.cumul_attack_timefrac.label = 'cumul_attack_timefrac'
        self.cumul_attack_timefrac.savename = filepath('attack_time_fraction_allpixels')

        self.frac_moderator_changes.desc_long = 'Fraction of active pixels changes issued from moderators'
        self.frac_moderator_changes.desc_short = 'fraction of moderator changes'
        self.frac_moderator_changes.savename = filepath('fraction_moderator_changes')

        self.frac_cooldowncheat_changes.desc_long = 'Fraction of active pixels changes that break the 5-minute per-user cooldown'
        self.frac_cooldowncheat_changes.desc_short = 'fraction of cooldown-cheating changes'
        self.frac_cooldowncheat_changes.label = 'frac_cooldowncheat_changes'
        self.frac_cooldowncheat_changes.savename = filepath('fraction_cooldowncheating_changes')

        self.frac_redundant_color_changes.desc_long = 'Fraction of active pixels changes that are redundant in color with the pixel content before its change'
        self.frac_redundant_color_changes.desc_short = 'fraction of redundant changes'
        self.frac_redundant_color_changes.label = 'frac_redundant_color_changes'
        self.frac_redundant_color_changes.savename = filepath('fraction_redundant_changes')

        self.frac_redundant_coloranduser_changes.desc_long = 'Fraction of active pixels changes that are redundant in color and user with the pixel content before its change'
        self.frac_redundant_coloranduser_changes.desc_short = 'fraction of redundant (color and user) changes'
        self.frac_redundant_coloranduser_changes.savename = filepath('fraction_redundant_coloranduser_changes')

