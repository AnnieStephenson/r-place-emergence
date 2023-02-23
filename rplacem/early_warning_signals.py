import numpy as np
import rplacem.canvas_part as cp
import rplacem.utilities as util
import rplacem.canvas_part_statistics as stat
import rplacem.variables_rplace2022 as var
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


    def calc_variance(self, x, time_window):
        '''
        calculates the variance vs time of the state variable
        '''
        x = pd.Series(x)
        variance = x.rolling(window=time_window).var()
        return np.array(variance)
        
    def calc_skewness(self, x, time_window):
        '''
        calculates the skewness vs time of the state variable
        '''
        x = pd.Series(x)
        skewness = x.rolling(window=time_window).skew()
        return np.array(skewness)

    def calc_autocorrelation(self, x, time_window):
        '''
        calculates the skewness vs time of the state variable
        '''
        x = pd.Series(x)
        autocorrelation = x.rolling(window=time_window).apply(lambda y: y.autocorr())
        return np.array(autocorrelation)

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
                ews[0,j,i,:] = self.calc_variance(x, self.time_window)
                ews[1,j,i,:] = self.calc_skewness(x, self.time_window)
                ews[2,j,i,:] = self.calc_autocorrelation(x, self.time_window)
        
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


def plot_ews_sing_var_offset(vari, alpha):
    n_comps = vari.shape[0]
    var_norm = np.zeros(vari.shape)
    for i in range(n_comps):
        var_norm[i,:] = vari[i,:]/np.nanmax(vari[i,:])
        plt.plot(var_norm[i,:], color=[0,0,0], alpha=alpha)
    var_mean = np.nanmean(var_norm, axis=0)
    plt.plot(var_mean/np.nanmax(var_mean))
    sns.despine()

def plot_ews_offset(ews_offset, 
                    state_offset,
                    state_ind,
                    alpha=0.03,
                    xlim=[0,100]):
 

    plt.figure()
    plot_ews_sing_var_offset(ews_offset[0,:,state_ind,:], alpha)
    plt.xlim(xlim)
    plt.ylabel('Variance (normalized)')
    plt.xlabel('Time bin')

    plt.figure()
    plot_ews_sing_var_offset(ews_offset[1,:,state_ind,:], alpha)
    plt.ylabel('Skewness (normalized)')
    plt.xlim(xlim)
    plt.xlabel('Time bin')

    plt.figure()
    plot_ews_sing_var_offset(ews_offset[2,:,state_ind,:], alpha)
    plt.ylabel('Autocorrelation (normalized)')
    plt.xlim(xlim)
    plt.xlabel('Time bin')

    plt.figure()
    plot_ews_sing_var_offset(state_offset[:,state_ind,:], alpha)
    plt.ylabel('State Variable (normalized)')
    plt.xlim(xlim)
    plt.xlabel('Time bin')

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


