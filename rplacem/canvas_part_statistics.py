import numpy as np
import os
import cv2
import rplacem.variables_rplace2022 as var
import rplacem.compute_variables as comp
import rplacem.utilities as util
import rplacem.transitions as trans
import shutil


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
            0.5: only for count_attack_defense_events(), only computes variables unrelated to transitions
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
            Set in comp.num_changes_and_users(), in count_attack_defense_events().

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
    n_t_bins_trans, t_interval_trans, t_ranges_trans : float
        parameters for time intervals, used exclusively in search_transitions()
            Set in __init__()

    TRANSITIONS
    transition_param : list of floats
        cutoff parameters for the trans.find_transitions() function;
        Beware, the last two arguments are in seconds and not in time intervals
    refimage_averaging_period : float
        time over which the reference (most stable) images before and after the transition are computed.

    VARIABLES
        STABILITY
    stability: float
        stability averaged over time and over all pixels of the CanvasPart.
            Needs compute_vars['mean_stability'] > 0
            Set in comp.stability(), in compute_mean_stability()
    stability_vst: 1d numpy array of floats
        stability averaged over all pixels of the CanvasPart, for each time bin.
            Needs compute_vars['stability'] > 0
            Set in comp.stability(), in compute_stability_vst()
    instability_vst_norm : 1d numpy array of floats
        Time-normalised instability: (1 - stability_vst) / t_norm
            Needs compute_vars['stability'] > 0
            Set in __init__()
    instability_for_trans : same as instability_vst_norm, but only to find transitions
    stable_image, second_stable_image, third_stable_image : pixels image info (2d numpy array containing color indices)
        Most stable image (and second and third most stable images) over the full active time range of the canvas part.
            Needs compute_vars['mean_stability'] > 1
            Set in comp.stability(), in compute_mean_stability()
    stable_image_vst, second_stable_image_vst, third_stable_image_vst :
        Same as stable_image, but as a 1d array of pixel images, of size n_t_bins
            Needs compute_vars['stability'] > 1
            Set in comp.stability(), in compute_stability_vst()
    diff_stable_pixels_vst, diff_stable_pixels_vst_norm : 1d numpy array of length n_t_bins
        Number of different pixels between the stable image over that time step
        and the stable image over the previous time step.
        diff_stable_pixels_vst_norm is normalized by the active area and t_norm
            Set in comp.stability(), in compute_stability_vst()
    diff_pixels_vst, diff_pixels_vst_norm : 1d numpy array of length n_t_bins
        same as above, but for the direct images at the end of the time step

        ENTROPY -- all need compute_vars['entropy'] > 0
    bmpsize : 1d numpy array of size n_t_bins
        Size of the bmp image (of the canvas part) file at each time step
            Needs compute_vars['entropy'] > 0
            Set in comp.save_part_over_time(), in compute_entropy()
    entropy_vst_bpmnorm : 1d numpy array of size n_t_bins
        Ratio of sizes of the png and bmp image files at each time step
            Needs compute_vars['entropy'] > 0
            Set in compute_entropy()
    entropy_vst : 1d numpy array of size n_t_bins
        Ratio of size of the png image file, to the number of active pixels, at each time step
            Set in compute_entropy()
    true_image_vst: list of size n_t_bins, of 2d numpy arrays.
        Each element is a pixels image info (2d numpy array containing color indices)
        True image of the canvas part at the center of the time interval
            Needs compute_vars['entropy'] > 1
            Set in comp.save_part_over_time(), in compute_entropy()

    TRANSITIONS -- all need compute_vars['transitions'] > 0
    refimage_pretrans, refimage_intrans, refimage_posttrans : array of size min(1, number of transitions), of 2d numpy arrays.
        Each element is a pixels image info (2d numpy array containing color indices)
        Reference stable image for the stable period (of duration refimage_averaging_period)
        before and after the transition, and at the time of the transition.
        If there is no detected transition, refimage_pretrans is set to self.stable_image
            Needs compute_vars['transitions'] > 1
            Set in trans.transition_and_reference_image(), in search_transitions()
    frac_diff_pixels_pre_vs_post_trans : 1d array of length (number of transitions)
        Fraction of the active pixels that differ between the pre- and post-transition stable images
            Needs compute_vars['transitions'] > 1
            Set in search_transitions()
    num_transitions : int
        Number of found transitions
            Set in search_transitions()
    transition_times: 2d array, shape (number of transitions, 6)
        Delimiting times for each transition.
        For each transition, is of the form
        [beg, end of pre stable period, beg, end of transition period, beg, end of post stable region]
            Set in trans.transition_and_reference_image(), in search_transitions()

    ATTACK-DEFENSE -- all need compute_vars['attackdefense'] > 0
    num_pixchanges : 1d array of length n_t_bins
        Number of pixel changes in each time bin
            Set in comp.num_changes_and_users(), in count_attack_defense_events()
    num_pixchanges_norm : same as num_pixchanges, normalized by t_norm and area_vst
    ratio_attdef_changes : 2d array of floats, shape (# transitions, n_t_bins)
        Ratio of attack pixel changes to the defense ones, in each time bin
            Set in count_attack_defense_events()
    frac_diff_pixels : 1d array of floats of length n_t_bins
        Fraction of the active pixels (area_vst) that differ between the
        pre- and post-transition stable images, in each time bin
            Set in count_attack_defense_events()
    num_users_total : float
        total number of user having contributed to the composition between tmin and tmax
    num_users_vst : 1d array of int of length n_t_bins
        number of users that changed any pixels in this composition in each time range
            Set in comp.num_changes_and_users(), in count_attack_defense_events()
    num_users_norm : same as num_users_vst, normalized by t_norm and area_vst
    frac_attackonly_users, frac_defenseonly_users, frac_bothattdef_users : 1d array of floats of length n_t_bins
        fraction of num_users that contributed pixels that are consistent (defenseonly) with
        refimage_pretrans, or not (attackonly), or fraction of users that did both (bothattdef)
            Set in count_attack_defense_events()
    ratio_attdef_changes_images_vst : array of size n_t_bins, of 2d numpy arrays.
        Each element is a pixels image info (2d numpy array containing color indices)
        Image containing, for each pixel, the ratio of attack to defense pixel changes, for each time step
    returntime_tbinned : 2d array of lists, shape (# transitions, n_t_bins)
        For each time bin, contains list of times for each freshly attacked pixel to recover to the ref image, thanks to a defense pixel change
    returntime_mean, returntime_median_overln2 : 2d array of shape (# transitions, n_t_bins)
        mean and median of returntime_tbinned in each time bin

    methods
    -------
    private:
        __init__
    protected:
    public:
        compute_stability_vst()
            Compute and set time-dependent stability attributes
        compute_mean_stability()
            Compute and set mean stability attributes
        compute_entropy()
            Compute and set images at each time step, and set related entropy attributes
        search_transitions()
            Look for transitions and set associated attributes
        count_attack_defense_events()
            Set attributes related to attack or defense changes or users, compared to a reference image
    '''

    def __init__(self,
                 cpart,
                 n_tbins=80,
                 tmax=var.TIME_TOTAL,
                 compute_vars={'stability': 3, 'mean_stability': 3, 'entropy': 3, 'transitions': 3, 'attackdefense': 2},
                 trans_param=[1e-2, 1.5e-3, 14400, 10800],
                 n_tbins_trans=150,
                 timeunit=300,  # 5 minutes
                 refimage_averaging_period=3600,  # 1 hour
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
        refimage_averaging_period :
            Time period over which the reference image is computed. Check class doc for details.
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
        n_t_bins_trans : float
            Fills n_t_bins_trans attribute. Check class doc for details.
        '''

        self.id = cpart.out_name()
        if compute_vars['stability'] < 3 and compute_vars['mean_stability'] < 3 and compute_vars['entropy'] < 3:
            renew = False
        dirpath = os.path.join(var.FIGS_PATH, str(cpart.out_name()))
        dir_exists = util.make_dir(dirpath, renew)

        self.compute_vars = compute_vars
        self.verbose = verbose
        self.area = len(cpart.coords[0])
        self.area_rectangle = cpart.width(0) * cpart.width(1)

        self.tmin = 0 # not functioning yet for tmin > 0
        self.tmax = tmax
        self.n_t_bins = n_tbins
        self.t_interval = (self.tmax - self.tmin) / n_tbins  # seconds
        self.t_unit = timeunit
        self.t_norm = self.t_interval / self.t_unit
        self.t_ranges = np.arange(self.tmin, tmax + self.t_interval - 1e-4, self.t_interval)

        self.n_t_bins_trans = n_tbins_trans
        self.t_interval_trans = (self.tmax - self.tmin) / n_tbins_trans  # seconds
        self.t_ranges_trans = np.arange(self.tmin, tmax + self.t_interval_trans - 1e-4, self.t_interval_trans)

        self.refimage_averaging_period = refimage_averaging_period

        self.instability_vst_norm = None
        self.instability_for_trans = None
        self.transition_param = trans_param
        self.refimage_pretrans = None
        self.num_transitions = None
        self.area_vst = None

        # mean stability
        self.compute_mean_stability(cpart, compute_vars['mean_stability'])

        # stability vs time
        self.compute_stability_vst(cpart, compute_vars['stability'])

        # entropy and images versus time. Better if self.area_vst is filled before (in num_changes_and_users)
        self.compute_entropy(cpart, compute_vars['entropy'])

        # find transitions. Needs that stability_vst is filled
        self.search_transitions(cpart, compute_vars['transitions'])
        # counting attack and defense users and pixel changes. Needs that refimage_pretrans is filled
        self.count_attack_defense_events(cpart, compute_vars['attackdefense'])

        # remove directory if it did not exist before and if dont_keep_dir
        if (not dir_exists) and dont_keep_dir:
            shutil.rmtree(dirpath)

    def compute_mean_stability(self, cpart, level):
        if level == 0:
            return
        if self.verbose:
            print('computing mean stability')

        tmin_all = max(self.tmin, np.min(cpart.border_path_times))
        tmax_all = min(self.tmax, np.max(cpart.border_path_times))
        res = comp.stability(cpart, np.asarray([0, tmin_all, tmax_all]), level > 1, level > 2, level > 3, True, self.verbose, None, None, self.t_unit)
        self.stability = res[0][1]
        self.stable_image = res[2][1] if level > 1 else None
        self.second_stable_image = res[3][1] if level > 1 else None
        self.third_stable_image = res[4][1] if level > 1 else None

    def compute_stability_vst(self, cpart, level):
        if level == 0:
            return
        if self.verbose:
            print('computing stability vs time')

        res = comp.stability(cpart, self.t_ranges, level > 1, level > 2, level > 3, True, self.verbose, None, None, self.t_unit)
        self.stability_vst = res[0]
        self.instability_vst_norm = res[1]
        self.stable_image_vst = res[2] if level > 1 else None
        self.second_stable_image_vst = res[3] if level > 1 else None
        self.third_stable_image_vst = res[4] if level > 1 else None
        self.diff_stable_pixels_vst = res[5]

        if level > 2:
            util.save_movie(os.path.join(var.FIGS_PATH, cpart.out_name(), 'VsTimeStab'), 15)

    def compute_entropy(self, cpart, level):
        if level == 0:
            return
        if self.verbose:
            print('computing entropy and images versus time')

        times = self.t_ranges - self.t_interval / 2
        times[0] = 0
        res = comp.save_part_over_time(cpart, times, level > 1, True, level < 3, False, self.verbose, True)
        size_bmp = res[0]
        size_png = res[1]
        num_active_pix = res[3]
        self.bmpsize = size_bmp[1:]
        self.true_image_vst = res[2] if level > 1 else None
        self.diff_pixels_vst = res[4][1:]

        self.entropy_vst_bpmnorm = size_png[1:] / size_bmp[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            self.entropy_vst = size_png[1:] / num_active_pix[1:]
        idx_dividebyzero = np.where(num_active_pix[1:] == 0)[0]
        self.entropy_vst[idx_dividebyzero] = self.entropy_vst_bpmnorm[idx_dividebyzero] * 3.2 # typical factor hard-coded here

        if level > 2:
            util.save_movie(os.path.join(var.FIGS_PATH, cpart.out_name(),'VsTime'), 15)

    def search_transitions(self, cpart, level):
        if level == 0:
            return
        if self.verbose:
            print('Searching transitions')

        # need instability vs time for the timeranges specific to transitions
        if self.instability_for_trans is None:
            if (not (self.instability_vst_norm is None) and self.n_t_bins == self.n_t_bins_trans):
                self.instability_for_trans = self.instability_vst_norm
            else:
                res = comp.stability(cpart, self.t_ranges_trans, False, False, False, True, self.verbose, self.t_unit)
                self.instability_for_trans = res[1]

        par = self.transition_param
        res = trans.transition_and_reference_image(cpart, self.t_ranges_trans, self.instability_for_trans,
                                                   level > 1, level > 2,
                                                   averaging_period=self.refimage_averaging_period,
                                                   cutoff=par[0],
                                                   cutoff_stable=par[1],
                                                   len_stable_intervals=int(par[2] / self.t_interval_trans) + 1,
                                                   dist_stableregion_transition=int(par[3] / self.t_interval_trans) + 1)

        self.refimage_pretrans = np.array(res[0]) if (level > 1 or self.compute_vars['attackdefense'] > 0) else None
        self.refimage_intrans = np.array(res[1]) if level > 1 else None
        self.refimage_posttrans = np.array(res[2]) if level > 1 else None
        self.frac_diff_pixels_pre_vs_post_trans = np.array(res[3]) / self.area
        self.transition_times = res[5]
        self.num_transitions = self.transition_times.shape[0]

    def count_attack_defense_events(self, cpart, level):
        if level == 0:
            return
        if self.verbose:
            print('counting attack and defense users and pixel changes')

        # need reference image
        if self.refimage_pretrans is None and level > 0.9:
            self.search_transitions(cpart, 1)
        # if no transition is detected (or level=0.5), take the most stable image over the whole active time as reference
        if level<0.9 or len(self.refimage_pretrans) == 0:
            if self.compute_vars['mean_stability'] < 2:
                self.compute_mean_stability(cpart, 2)
            self.refimage_pretrans = np.expand_dims(self.stable_image, axis=0)

        num_trans_nonzero = 1 if level<1 else np.max([self.num_transitions, 1])

        self.num_pixchanges = np.zeros((num_trans_nonzero, self.n_t_bins))
        self.num_pixchanges_norm = np.zeros((num_trans_nonzero, self.n_t_bins))
        self.num_users_vst = np.zeros((num_trans_nonzero, self.n_t_bins))
        self.num_users_norm = np.zeros((num_trans_nonzero, self.n_t_bins))
        if level > 0.9:
            self.ratio_attdef_changes = np.zeros((num_trans_nonzero, self.n_t_bins))
            self.frac_diff_pixels = np.zeros((num_trans_nonzero, self.n_t_bins))
            self.frac_attackonly_users = np.zeros((num_trans_nonzero, self.n_t_bins))
            self.frac_defenseonly_users = np.zeros((num_trans_nonzero, self.n_t_bins))
            self.frac_bothattdef_users = np.zeros((num_trans_nonzero, self.n_t_bins))
            self.returntime_tbinned = np.empty((num_trans_nonzero, self.n_t_bins), dtype=object)
            self.returntime_mean = np.zeros((num_trans_nonzero, self.n_t_bins))
            self.returntime_median_overln2 = np.zeros((num_trans_nonzero, self.n_t_bins))

        if level > 1:
            self.ratio_attdef_changes_images_vst = np.zeros((num_trans_nonzero, self.n_t_bins,
                                                             self.refimage_pretrans[0].shape[0], self.refimage_pretrans[0].shape[1]))

        if self.verbose:
            print('run comp.num_changes_and_users in loop over transitions')
        for i in range(num_trans_nonzero):
            refimage = self.refimage_pretrans[i]
            res = comp.num_changes_and_users(cpart, self.t_ranges, refimage, level > 1.5, level > 2.5, level <= 0.9)
            self.num_pixchanges[i,:] = res[0]
            self.num_users_total = res[9] # same for all transitions
            self.area_vst = res[3]
            self.num_users_vst[i,:] = res[8]

            if level > 1.5:
                self.ratio_attdef_changes_images_vst[i,:,:,:] = res[10]

            with np.errstate(divide='ignore', invalid='ignore'):
                self.num_pixchanges_norm[i,:] = res[0] / self.t_norm / self.area_vst # res[3] is the *time-dependent* active area
                self.num_users_norm[i,:] = res[8] / self.t_norm / self.area_vst
                self.diff_stable_pixels_vst_norm = self.diff_stable_pixels_vst / self.t_norm / self.area_vst
                self.diff_pixels_vst_norm = self.diff_pixels_vst / self.t_norm / self.area_vst
                if level > 0.9:
                    self.ratio_attdef_changes[i,:] = res[2] / res[1]
                    self.frac_diff_pixels[i,:] = res[4] / self.area_vst
                    self.frac_attackonly_users[i,:] = res[5] / res[9]
                    self.frac_defenseonly_users[i,:] = res[6] / res[9]
                    self.frac_bothattdef_users[i,:] = res[7] / res[9]

            self.num_pixchanges_norm[-1][np.where(self.area_vst == 0)] = 0.
            self.num_users_norm[-1][np.where(self.area_vst == 0)] = 0.
            if level > 0.9:
                self.ratio_attdef_changes[-1][np.where((res[1] == 0) & (res[2] > 0))] = 100.
                self.ratio_attdef_changes[-1][np.where((res[1] == 0) & (res[2] == 0))] = 1.
                self.frac_diff_pixels[-1][np.where(self.area_vst == 0)] = 0.
                self.diff_stable_pixels_vst_norm[np.where(self.area_vst == 0)] = 0.
                self.diff_pixels_vst_norm[np.where(self.area_vst == 0)] = 0.
                self.frac_attackonly_users[-1][np.where(res[9] == 0)] = 0.5
                self.frac_defenseonly_users[-1][np.where(res[9] == 0)] = 0.5
                self.frac_bothattdef_users[-1][np.where(res[9] == 0)] = 0.

                returnt = res[11]
                time_newattack = res[12]
                timebin_ind = np.digitize(time_newattack, self.t_ranges) - 2  # TODO: check this
                for j in range(0, self.n_t_bins):  # initialize with empty lists
                    self.returntime_tbinned[i][j] = []
                for j in range(0, len(returnt)):  # fill the histogram-like array using the result from np.digitize
                    self.returntime_tbinned[i][timebin_ind[j]].append(returnt[j])
                for j in range(0, self.n_t_bins):  # cannot use more clever numpy because the lists in axis #3 are of different sizes
                    self.returntime_mean[i, j] = np.mean(np.array(self.returntime_tbinned[i, j]))
                    self.returntime_median_overln2[i, j] = np.median(np.array(self.returntime_tbinned[i, j]))

        if level > 0.9:
            self.returntime_median_overln2 /= np.log(2)

        if level > 2:
            util.save_movie(os.path.join(var.FIGS_PATH, cpart.out_name(), 'attack_defense_ratio'), 15)
