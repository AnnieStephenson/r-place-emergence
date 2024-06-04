import numpy as np
import os
import rplacem.canvas_part as cp
import rplacem.utilities as util
import rplacem.canvas_part_statistics as stat
from rplacem import var as var
import rplacem.transitions as trans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib.colors as pltcolors

class EarlyWarnSignals(object):
    '''
    object containing all the EWS stuff for a given canvasPart object

    attributes
    ----------

    methods
    -------
    '''

    def __init__(self,
                 cpart_stat,
                 time_window,
                 state_vars=None,
                 half_stab_interval=True,
                 thresh_low = 0.01,
                 thresh_high = 20,
                 ):

        # set some basic params
        self.time_window = time_window
        self.half_stab_interval = half_stab_interval
        self.thresh_high = thresh_high,
        self.thresh_low = thresh_low
        self.num_ews_vars = 3
        self.trans_ind = 0 # for now, we are only taking the first transition
        if state_vars==None:
            self.num_state_vars = 13
        else:
            self.num_state_vars = len(state_vars)
            
        # calculate and set the early warning signals
        ews, state_vars = self.calc_ews(cpart_stat, state_vars=state_vars)
        self.ews = ews
        self.state_vars = state_vars

        # calculate the pre-transition mean ews values
        ews_mean_stab = self.calc_mean_stab_ews(cpart_stat)
        self.ews_mean_stab = ews_mean_stab

        # classify and set the ews types
        (ews_type1, 
         ews_type2,
         ews_type3) = self.classify_trans_ews(cpart_stat)

        self.ews_type1 = ews_type1
        self.ews_type2 = ews_type2
        self.ews_type3 = ews_type3


    def calc_ews(self, canvas_part_stat, state_vars = None):
        '''    
        calculates all the EWS vs time for all the state variables 
        and all the transitions for a single canvas part
        
        time window is currently in units of time_interval, need to fix
        '''
        
        if state_vars == None:
            state_vars = [canvas_part_stat.entropy_vst,
                        canvas_part_stat.entropy_vst_bpmnorm,
                        canvas_part_stat.stability_vst,
                        canvas_part_stat.instability_vst_norm,
                        canvas_part_stat.num_pixchanges,
                        canvas_part_stat.num_pixchanges_norm,
                        canvas_part_stat.ratio_attdef_changes,
                        canvas_part_stat.frac_diff_pixels,
                        canvas_part_stat.num_users_vst,
                        canvas_part_stat.num_users_norm,
                        canvas_part_stat.frac_attackonly_users, 
                        canvas_part_stat.frac_defenseonly_users, 
                        canvas_part_stat.frac_bothattdef_users]
            
        
        num_trans_nonzero = np.max([1, canvas_part_stat.num_transitions])
        ews = np.zeros((self.num_ews_vars, num_trans_nonzero, len(state_vars),  canvas_part_stat.n_t_bins))
        
        for i in range(len(state_vars)):
            for j in range(num_trans_nonzero):
                if len(state_vars[i].shape)==2:
                    x = state_vars[i][j]
                else: 
                    x = state_vars[i]
                ews[0,j,i,:] = calc_variance(x, self.time_window)
                ews[1,j,i,:] = calc_skewness(x, self.time_window)
                ews[2,j,i,:] = calc_autocorrelation(x, self.time_window)
        
        return ews, state_vars  

    def calc_mean_stab_ews(self, canvas_comp_stat):

        # find the index where the stable time starts and ends
        t_pretrans_stable_start = canvas_comp_stat.transition_times[self.trans_ind,0]
        t_pretrans_stable_end = canvas_comp_stat.transition_times[self.trans_ind,1]
        if self.half_stab_interval:
            t_pretrans_stable_end = t_pretrans_stable_start + (t_pretrans_stable_end-t_pretrans_stable_start)/2

        # find the time indices for when to take the mean value
        ind_stab0 = np.argmin(np.abs(t_pretrans_stable_start-canvas_comp_stat.t_ranges))
        ind_stab1 = np.argmin(np.abs(t_pretrans_stable_end-canvas_comp_stat.t_ranges))
        
        # get the mean ews vars for that time
        ews_mean_stab = np.mean(self.ews[:,self.trans_ind, :, ind_stab0:ind_stab1], axis=2)

        return ews_mean_stab

    def classify_trans_ews(self,
                           cpart_stat):
        t_pretrans_stable_end = cpart_stat.transition_times[self.trans_ind, 1]
        t_trans_start = cpart_stat.transition_times[self.trans_ind, 2]
        if self.half_stab_interval:
            t_pretrans_stable_start = cpart_stat.transition_times[self.trans_ind, 0]
            t_pretrans_stable_end = t_pretrans_stable_start + (t_pretrans_stable_end-t_pretrans_stable_start)/2
        
        # find time indices for when to apply the classification check
        ind1 = np.argmin(np.abs(t_pretrans_stable_end - cpart_stat.t_ranges))
        ind2 = np.argmin(np.abs(t_trans_start - cpart_stat.t_ranges))  

        ews_type1 = self.ews[:, self.trans_ind, :, ind1:ind2]>=self.thresh_high*self.ews_mean_stab[:, :, np.newaxis]
        ews_type1 = np.sum(ews_type1, axis=2)
        ews_type1[ews_type1>0] = 1
        
        ews_type2 = self.ews[:, self.trans_ind, :, ind1:ind2]<=self.thresh_low*self.ews_mean_stab[:, :, np.newaxis]
        ews_type2 = np.sum(ews_type2, axis=2)
        ews_type2[ews_type2>0] = 1
        
        ews_type3 = np.logical_and(ews_type1, ews_type2).astype(int)
        
        return ews_type1, ews_type2, ews_type3 
    

####### OUTSIDE CLASS #############

def calc_ews_shift(canvas_comp_stat_list, early_warn_signals_list,
                   time_padding=20):
    '''
    calculates the ews for a list of CanvasPartStatistics objects,
    and offsets their times so that the pre-transition stable period
    is the first point in the time series for each canvas_part
    
    Currently only uses the first transition for a given composition
    '''

    # set some variables assuming all are the same in list
    n_tbins = canvas_comp_stat_list[0].n_t_bins 
    trans_ind = early_warn_signals_list[0].trans_ind
    n_state_vars = early_warn_signals_list[0].num_state_vars 
    n_ews_vars = early_warn_signals_list[0].num_ews_vars
    time_window = early_warn_signals_list[0].time_window
    n_comps = len(early_warn_signals_list)
    
    ews_offset = np.zeros((n_ews_vars, n_comps, n_state_vars, n_tbins))
    state_offset = np.zeros((n_comps, n_state_vars, n_tbins))
    trans_time_diffs = np.zeros((n_comps,5))
    for i in range(len(early_warn_signals_list)):
        print('CanvasComp count: ' + str(i + 1))
        
        # get relevant transition times and their indices
        trans_time_diffs[i,:] = np.diff(canvas_comp_stat_list[i].transition_times[0])
        t_ranges = canvas_comp_stat_list[i].t_ranges
        t_trans_start = canvas_comp_stat_list[i].transition_times[0][2]
        t_post_stable_start = canvas_comp_stat_list[i].transition_times[0][4]

        # find the time indices for determining the offsett
        ind1 = np.argmin(np.abs(t_ranges - t_trans_start))-time_padding
        ind2 = np.argmin(np.abs(t_ranges - t_post_stable_start))+time_padding
        len_trans_time = ind2-ind1
                        
        # get the shifted ews for each state variable
        ews_offset[:,i,:,0:len_trans_time] = early_warn_signals_list[i].ews[:,trans_ind,:,ind1:ind2]
        state_vars = early_warn_signals_list[i].state_vars
        for j in range(len(state_vars)):
            if len(state_vars[j].shape)==2:
                state_vars[j] = state_vars[j][0,:]
            state_offset[i,j,0:len_trans_time] =  state_vars[j][ind1:ind2]
            
    return ews_offset, state_offset, trans_time_diffs


def plot_ews_sing_var_offset(time, vari, alpha, mean_linewidth=2.):
    n_comps = vari.shape[0]
    var_norm = np.zeros(vari.shape)
    for i in range(n_comps):
        var_norm[i,:] = vari[i,:]/np.nanmax(vari[i,:])
        plt.plot(time, var_norm[i,:], color=[0,0,0], alpha=alpha)
    var_mean = np.nanmean(var_norm, axis=0)
    plt.plot(time, var_mean, linewidth=mean_linewidth)
    sns.despine()

def plot_ews_offset(time,
                    ews_offset, 
                    state_offset,
                    state_ind,
                    alpha=0.03,
                    xlim=[0,100],
                    save=False,
                    mean_linewidth = 2.,
                    figsize=(8,5),
                    fontsize=10,
                    state_var_label='State Variable (normalized)',
                    xlabel='Time bins'):
 

    plt.figure(figsize=figsize)
    plot_ews_sing_var_offset(time, ews_offset[0,:,state_ind,:], alpha, mean_linewidth=mean_linewidth)
    plt.xlim(xlim)
    plt.ylabel('Variance', fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.yticks([0, .2, .4, .6, 0.8])
    plt.ylim([0,1])
    if save:
        plt.tight_layout()
        plt.savefig('variance.png', dpi=600)

    plt.figure(figsize=figsize)
    plot_ews_sing_var_offset(time, ews_offset[1,:,state_ind,:], alpha, mean_linewidth=mean_linewidth)
    plt.ylabel('Skewness',fontsize=fontsize)
    plt.yticks([-1, -.5, 0, 0.5])
    plt.ylim([-1,1])
    plt.xlim(xlim)
    plt.xlabel(xlabel, fontsize=fontsize)
    if save:
        plt.tight_layout()
        plt.savefig('skewness.png', dpi=600)

    plt.figure(figsize=figsize)
    plot_ews_sing_var_offset(time, ews_offset[2,:,state_ind,:], alpha, mean_linewidth=mean_linewidth)
    plt.ylabel('Autocorrelation',fontsize=fontsize)
    plt.yticks([-1, -0.5, 0, .5])
    plt.ylim([-1,1])
    plt.xlim(xlim)
    plt.xlabel(xlabel,fontsize=fontsize)
    if save:
        plt.tight_layout()
        plt.savefig('autocorr.png', dpi=600)

    plt.figure(figsize=figsize)
    plot_ews_sing_var_offset(time, state_offset[:,state_ind,:], alpha, mean_linewidth=mean_linewidth)
    plt.ylabel(state_var_label,fontsize=fontsize)
    plt.xlim(xlim)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.yticks([0, .2, .4, .6, 0.8])
    plt.ylim([0,1])
    if save:
        plt.tight_layout()
        plt.savefig('state_var.png', dpi=600)

def find_ews_type_list_inds(early_warn_signal_list,
                            var_ind,
                            state_ind):
    type1_inds = []
    type2_inds = []
    type3_inds = []
    for i in range(len(early_warn_signal_list)):
        ews_type1 = early_warn_signal_list[i].ews_type1[var_ind,state_ind]
        ews_type2 = early_warn_signal_list[i].ews_type2[var_ind,state_ind]
        ews_type3 = early_warn_signal_list[i].ews_type3[var_ind,state_ind]

        if ews_type1==1:
           type1_inds.append(i)
        if ews_type2==1:
           type2_inds.append(i)
        if ews_type3==1:
           type3_inds.append(i)

    return np.array(type1_inds), np.array(type2_inds), np.array(type3_inds)

def plot_ews_offset_types(ews_offset, 
                          state_offset,
                          early_warn_signal_list, 
                          state_ind,
                          ews_var_ind,
                          alpha=0.03,
                          xlim=[0,100]):
    labels = ['Variance (normalized)',
              'Skewness (normalized)',
              'Autocorrelation (normalized)']

    (type1_inds, 
    type2_inds, 
    type3_inds) = find_ews_type_list_inds(early_warn_signal_list, ews_var_ind,state_ind)
    
    if len(type1_inds>0):
        plt.figure()
        plot_ews_sing_var_offset(ews_offset[ews_var_ind, type1_inds, state_ind,:], alpha)
        plt.xlim(xlim)
        plt.ylabel(labels[ews_var_ind])
        plt.xlabel('Time bin')
        plt.title('type 1')

    if len(type2_inds>0):
        plt.figure()
        plot_ews_sing_var_offset(ews_offset[ews_var_ind, type2_inds, state_ind,:], alpha)
        plt.xlim(xlim)
        plt.ylabel(labels[ews_var_ind])
        plt.xlabel('Time bin')
        plt.title('type 2')

    if len(type3_inds>0):
        plt.figure()
        plot_ews_sing_var_offset(ews_offset[ews_var_ind, type3_inds, state_ind,:], alpha)
        plot_ews_sing_var_offset(state_offset[ews_var_ind, type3_inds, state_ind,:], alpha)
        plt.xlim(xlim)
        plt.ylabel(labels[ews_var_ind])
        plt.xlabel('Time bin')
        plt.title('type 3')

def ratio_to_slidingmean(ewsvar, tint, slidingrange=36000):
    '''Ratio of value of [ewsvar] at time t to the average of [ewsvar] over the range [t, t-slidingrange] '''
    sliding_indrange = math.ceil(slidingrange / tint)
    # mean over sliding window
    mean_slided = np.array( pd.Series(ewsvar).rolling(window=sliding_indrange).mean() )
    # average over existing points when t<slidingrange
    for i in range(1, sliding_indrange):
        mean_slided[i] = np.mean(ewsvar[:i+1])
    # want to look at the past only, so offset of 1 time index
    mean_slided = np.roll(mean_slided, 1)
    mean_slided[0] = ewsvar[0] # ratio =1 for first value

    # protection against zero mean
    with np.errstate(divide='ignore', invalid='ignore'):
        res = ewsvar / mean_slided
    res[mean_slided == 0] = 1

    return res

def firing_times(cpstat, ewsvar, earlyness, thres, slidingrange=21600, warning_cooldown=14400):
    '''Computes the times at which a given ews is firing.
    Returns the times of good firings and of all firings (including false positives), 
    and the ratio of their number.
    
    ewsvar, cpstat: ews variable and CanvasPartStatistics object to be tested
    earlyness: how far away from the beginning of the transition must the warning be
    thres: threshold on the ratio of the input variable [ewsvar] to its preceding sliding average
        to fire a warning
    slidingrange: range on which ewsvar is averaged before the current value
    warning_cooldown: both the min distance between two warnings, 
        and the max distance between a good warning and transition_time
    '''

    tint = cpstat.t_interval
    numtrans = cpstat.num_transitions
    slidingrange_ind = math.ceil(slidingrange / tint)
    # exclude first slidingrange hours
    exclude_beg = np.arange(0, slidingrange_ind)
    # exclude times with no active pixels, plus a safe slidingrange time afterwards
    exclude_inactive = np.where(cpstat.area_vst == 0)[0]
    if len(exclude_inactive) == 0:
        exclude_afterinactive = np.array([])
    else:  
        last_inactive = np.max(exclude_inactive)
        exclude_afterinactive = last_inactive + 1 + exclude_beg
        exclude_afterinactive = np.minimum(exclude_afterinactive, cpstat.n_t_bins - 1) # cap values at length of t_ranges
    # exclude white only times
    exclude_end = np.arange(math.floor((var.TIME_WHITEOUT - earlyness) / tint), math.ceil(var.TIME_TOTAL / tint))
    # exclude time within [earlyness] time before transition, and [slidingrange] time after transition
    _, mean_trans_starttime, median_trans_starttime, _ = trans.transition_start_time(cpstat)
    exclude_trans = []
    for tr in range(0, numtrans):
        exclude_trans.append( np.arange(math.floor((median_trans_starttime[tr] - earlyness) / tint), 
                                        math.ceil((cpstat.transition_times[tr][3] + slidingrange) / tint)) )
    # gather all excluded indices
    exclude_all_trans = np.concatenate(exclude_trans)
    exclude_inds = np.unique(np.concatenate((exclude_beg, exclude_end, exclude_inactive, exclude_afterinactive, exclude_all_trans)))
    mask_inds = np.full((len(ewsvar)), True, dtype=np.bool_)
    mask_inds[np.array(exclude_inds, dtype=int)] = False

    # get time indices at which the threshold is exceeded 
    ratio_to_mean = ratio_to_slidingmean(ewsvar, tint, slidingrange)
    if thres > 1:
        fire_timeind = np.where((ratio_to_mean > thres) & mask_inds)[0]
    else:
        fire_timeind = np.where((ratio_to_mean < thres) & mask_inds)[0]
    fire_timeind = np.sort(fire_timeind)
    # count how many ranges of size [warning_cooldown] contain at least 1 firing, followed or not by a transition
    prev_warns = []
    prev_goodwarns = []
    for t in fire_timeind:
        fire_t = cpstat.t_ranges[t]
        # reject firings that are too close to a previous warning
        iswarn = (len(prev_warns) == 0) or (fire_t >= prev_warns[-1] + warning_cooldown)
        if iswarn:
            prev_warns.append(fire_t) 

            # is this a good warning? accept times between transt-cooldown and transt-earlyness
            for trans_t in median_trans_starttime:
                if trans_t - warning_cooldown < fire_t and fire_t <= trans_t - earlyness:
                    prev_goodwarns.append(fire_t) 
                    break
            
    ratio_out = len(prev_goodwarns) / len(prev_warns) if len(prev_warns) > 0 else 0
    return ratio_out, prev_warns, prev_goodwarns

def ews_2Dsignificance_1comp(cpstat, ewsvar, val_thres, val_earlyness, warning_cooldown, vname=''):

    num_val_earlyness = len(val_earlyness)
    num_val_thres = len(val_thres)
    signif = np.zeros((num_val_thres, num_val_earlyness))
    for thres_ind in np.arange(0, num_val_thres):
        for earlyn_ind in np.arange(0, num_val_earlyness):
            thres = val_thres[thres_ind]
            earlyn = val_earlyness[earlyn_ind]
            signif[thres_ind, earlyn_ind] = firing_times(cpstat, ewsvar, earlyn, thres, slidingrange=21600, warning_cooldown=warning_cooldown)[0]
            #print(thres, earlyn, signif[thres_ind, earlyn_ind])

    if vname != '':
        # plot 2d significance
        plt.pcolormesh( val_earlyness, val_thres, signif, 
                        cmap=plt.cm.jet)#, norm=pltcolors.LogNorm(vmin=0.95, vmax=700))
        plt.ylabel('threshold on ratio to preceding sliding mean')
        plt.xlabel('earlyness [s]')
        plt.yscale('log')
        plt.xscale('log')
        #plt.ticklabel_format(style='scientific')
        plt.colorbar(label='# good warnings / # total warnings')
        plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'EWS_significance_'+vname+'.png'))

    return signif

def ews_2Dsignificance_allcomp(cpstats, warning_cooldown = 14400, ews_slidingwindow=4000, singlecompsave=False):
    
    tint = cpstats[0].t_interval # time rnages should be the same for all compos
    sliding_window = math.ceil(ews_slidingwindow / tint)

    # define threshold and earlyness ranges of values
    num_val_thres = 20
    val_thres1 = np.logspace(-2, 0, int(num_val_thres/2)) # log binning from 1e-2 to 1e3
    val_thres2 = np.logspace(0, 2.5, int(num_val_thres/2) + 1)[1:]
    val_thres = np.concatenate((val_thres1, val_thres2))
    val_earlyness = np.hstack((np.array([tint / 2]),
                               np.arange(tint, 5*tint, tint),
                               np.arange(5*tint, 7200, 2*tint),
                               np.arange(7200, warning_cooldown, 1800)))

    varnames = [
                'number_pixelchanges',
                'number_pixelchanges_variance',
                'number_pixelchanges_skewness',
                'number_pixelchanges_autocorrelation',
                'ratio_attacktodefense_pixelchanges',
                'number_user',
                'number_user_variance',
                'number_user_skewness',
                'number_user_autocorrelation',
                'fraction_users_onlyattacking',
                'median_returntime',
                'instability',
                'number_pixels_differing_from_previoustime',
                'entropy'
                ]
    numvar = len(varnames)
    
    # 2D significance averaged over all compos, for each variable
    signif = np.zeros((numvar, len(val_thres), len(val_earlyness)))

    # Loop on CanvasPartStatistics objects
    num_cpstat = 0
    for cpstat in cpstats:
        if cpstat.num_transitions == 0 or cpstat.num_transitions is None or cpstat.area < 200:
            continue
        num_cpstat += 1
        print(num_cpstat, cpstat.id)
        # TODO: should think about how to take into account these variables using another transition than the first
        tr = 0
        vars = [cpstat.num_pixchanges_norm[tr],
                calc_variance(cpstat.num_pixchanges_norm[tr], sliding_window),
                calc_skewness(cpstat.num_pixchanges_norm[tr], sliding_window),
                calc_autocorrelation(cpstat.num_pixchanges_norm[tr], sliding_window),
                cpstat.ratio_attdef_changes[tr],
                cpstat.num_users_norm[tr],
                calc_variance(cpstat.num_users_norm[tr], sliding_window),
                calc_skewness(cpstat.num_users_norm[tr], sliding_window),
                calc_autocorrelation(cpstat.num_users_norm[tr], sliding_window),
                cpstat.frac_attackonly_users[tr],
                cpstat.returntime_median_overln2[tr],
                cpstat.instability_vst_norm,
                cpstat.diff_stable_pixels_vst_norm,
                cpstat.entropy_vst
                ]
        for iv in range(0, numvar):
            v = vars[iv]
            #print(varnames[iv])
            signif[iv] += ews_2Dsignificance_1comp(cpstat, v, val_thres, val_earlyness, warning_cooldown)
    
    signif /= num_cpstat

    for iv in range(0, numvar):
        vname = varnames[iv]
        # plot 2d significance
        plt.clf()
        plt.pcolormesh( val_earlyness, val_thres, signif[iv], 
                        cmap=plt.cm.jet)#, norm=pltcolors.LogNorm(vmin=0.95, vmax=700))
        plt.ylabel('threshold on ratio to preceding sliding mean')
        plt.xlabel('earlyness [s]')
        plt.yscale('log')
        plt.xscale('log')
        #plt.ticklabel_format(style='scientific')
        plt.colorbar(label='average_all_compos( # good warnings / # total warnings )')
        if singlecompsave:
            util.make_dir(os.path.join(var.FIGS_PATH, cpstat.id))
            path = os.path.join(var.FIGS_PATH, cpstat.id, 'EWS_significance_'+vname+'.png')
        else:
            path = os.path.join(var.FIGS_PATH, 'EWS_significance_'+vname+'.png')
        plt.savefig(path)

    return signif, varnames