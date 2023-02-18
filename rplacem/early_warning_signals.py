import numpy as np
import rplacem.canvas_part as cp
import rplacem.utilities as util
import rplacem.canvas_part_statistics as stat
import rplacem.variables_rplace2022 as var
import pandas as pd

def calc_variance(x, time_window):
    '''
    calculates the variance vs time of the state variable
    '''
    x = pd.Series(x)
    variance = x.rolling(window=time_window).var()
    return np.array(variance)
    
def calc_skewness(x, time_window):
    '''
    calculates the skewness vs time of the state variable
    '''
    x = pd.Series(x)
    skewness = x.rolling(window=time_window).skew()
    return np.array(skewness)

def calc_autocorrelation(x, time_window):
    '''
    calculates the skewness vs time of the state variable
    '''
    x = pd.Series(x)
    autocorrelation = x.rolling(window=time_window).apply(lambda y: y.autocorr())
    return np.array(autocorrelation)

def calc_ews(canvas_part_stat, time_window, state_vars = None):
    '''    
    calculates all the EWS vs time for all the state variables 
    and all the transitions for a single canvas part
    
    time window is currently in units of time_interval, need to fix
    '''
    num_ews = 3 # so far
    
    if state_vars == None:
        state_vars = [canvas_part_stat.entropy_vst,
                      canvas_part_stat.entropy_vst_bpmnorm,
                      canvas_part_stat.stability_vst,
                      canvas_part_stat.num_pixchanges]
    
    ews = np.zeros((3, canvas_part_stat.num_transitions, len(state_vars),  canvas_part_stat.n_t_bins))
    
    for i in range(len(state_vars)):
        for j in range(canvas_part_stat.num_transitions):
            if len(state_vars[i].shape)==2:
                x = state_vars[i][j]
            else: 
                x = state_vars[i]
            ews[0,j,i,:] = calc_variance(x, time_window)
            ews[1,j,i,:] = calc_skewness(x, time_window)
            ews[2,j,i,:] = calc_autocorrelation(x, time_window)
    
    return ews  