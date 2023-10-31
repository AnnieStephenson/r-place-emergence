import numpy as np
import pickle
import os
import rplacem.globalvariables_peryear as vars
var = vars.var
import rplacem.transitions as tran
import math
import gc

# grab canvas_part_statistics object
file_path = os.path.join(var.DATA_PATH, 'canvas_composition_statistics_all.pickle') 
with open(file_path, 'rb') as f:
    cpstat0 = pickle.load(f)[0]
    f.close()
    del f
gc.collect()

# Define times that are kept before each event used in 
tstep = cpstat0.t_interval
trans_param = cpstat0.transition_param
watchrange_min = max(trans_param[2] + trans_param[3], 6*3600) # minimal time range kept for single time series instances in training

watch_timeidx_diff = np.array([1, 1, 1, 2, 3, 5, 8, 10, 12, 15, 15]) #np.array([1, 1, 2, 5, 13, 17, 17, 17]) # time indices kept, preceding the time of an event 
while tstep * (np.sum(watch_timeidx_diff)-1) < watchrange_min:
    watch_timeidx_diff = np.append(watch_timeidx_diff, 12)
watch_timeidx = - np.flip(np.cumsum(watch_timeidx_diff)) + 1 # starts at -max_watch_index and ends at 0
print(watch_timeidx)
n_traintimes = watch_timeidx.shape[0] # number of values in time kept for each variable of each event
watch_timeindrange = np.empty(n_traintimes, dtype=object) # 1d array of 1d lists of indices that will be averaged over
for i in range(0, n_traintimes):
    watch_timeindrange[i] = np.arange(watch_timeidx[i], ( 1 if (i == n_traintimes-1) else watch_timeidx[i+1] ))
len_watchrange = - watch_timeidx[0] # length in indices
watchrange = tstep * len_watchrange # length in seconds
print(watchrange)

# Same but for coarse variables
watch_timeidx_diff_coarse = np.array([1, 3, 8, 9, 10, 12, 15, 15])#np.array([1, 8, 13, 17, 17, 17]) # time indices kept. For smoothened variables
while tstep * (np.sum(watch_timeidx_diff_coarse)-1) < watchrange_min:
    watch_timeidx_diff_coarse = np.append(watch_timeidx_diff_coarse, 12)
watch_timeidx_coarse = - np.flip(np.cumsum(watch_timeidx_diff_coarse)) + 1
print(watch_timeidx_coarse)
n_traintimes_coarse = watch_timeidx_coarse.shape[0]
watch_timeindrange_coarse = np.empty(n_traintimes_coarse, dtype=object)
for i in range(0, n_traintimes_coarse):
    watch_timeindrange_coarse[i] = np.arange(watch_timeidx_coarse[i], ( 1 if (i == n_traintimes_coarse-1) else watch_timeidx_coarse[i+1] ))
if (- watch_timeidx_coarse[0]) != len_watchrange:
    ValueError('watchrange length is different for coarse and non-coarse!')

def variables_from_cpstat(cps):
    '''
    Array of cpstat time series variables that are kept for training/evaluation.
    The first boolean array says if the ratio of the 
    variable to its average on the preceding sliding window is taken.
    '''
    cps.entropy.set_ratio_to_sw_average()
    cps.fractal_dim_weighted.set_ratio_to_sw_average()

    take_ratio_to_average = np.array([0,0,0,0,0,0,0,0,0,1,1,0,0,0], dtype=bool)
    coarse_timeranges = np.array([0,0,0,0,0,1,1,1,0,0,0,0,1,0], dtype=bool)
    vars = np.array([cps.frac_pixdiff_inst_vs_swref,
                     cps.frac_pixdiff_inst_vs_stable_norm,
                     cps.frac_attack_changes,
                     cps.n_changes_norm,
                     cps.instability_norm,
                     cps.n_users_sw_norm,
                     cps.returntime_percentile90_overln2,
                     cps.returntime_median_overln2,
                     #cps.returntime_mean,
                     cps.frac_redundant_color_changes,
                     cps.entropy,
                     cps.fractal_dim_weighted,
                     cps.frac_users_new_vs_sw,
                     cps.changes_per_user_sw,
                     cps.cumul_attack_timefrac
                    ])
    if len(take_ratio_to_average) != len(vars) or len(coarse_timeranges) != len(vars):
        raise ValueError('ERROR: vars and take_ratio_to_average and coarse_timeranges must be of same length!')
    
    return(vars, take_ratio_to_average, coarse_timeranges)

cpstatvars = variables_from_cpstat(cpstat0)
coarse_timerange = cpstatvars[2]
n_cpstatvars = np.count_nonzero(coarse_timerange == 0)
n_cpstatvars_coarse = np.count_nonzero(coarse_timerange == 1)
n_trainingvars = n_cpstatvars * n_traintimes + n_cpstatvars_coarse * n_traintimes_coarse


def get_vals_from_var(v, tidx, coarse=False):
    '''
    Return values of the variable at all preceding watch time indices [tidx - watch_timeidx]
    Variables are averaged over values when watch time indices have a separation > 1
    '''
    n_times = n_traintimes_coarse if coarse else n_traintimes
    vals = np.empty(n_times, dtype=np.float64)
    for i in range(0, n_times):
        inds = tidx + (watch_timeindrange_coarse if coarse else watch_timeindrange)[i]
        vals[i] = np.mean(v[inds])
    return vals

def get_earliness(cpstat, t, trans_starttimes):
    '''
    Return the earliness of a potential signal at time t compared to the transitions present in cpstat.
    A random high (ie above 10h) earliness value is given when there is no transition in the composition
    '''
    min_earliness_notrans = 10*3600 # minimum random earliness given to events in no-transition compositions
    max_earliness = var.TIME_WHITEOUT - max(cpstat.sw_width_sec, watchrange) # max range of earliness

    if cpstat.n_transitions == 0: # no transitions
        # keep continuity of earliness values even here, but do not give value lower than some minimum
        return var.TIME_WHITEOUT - t + min_earliness_notrans
    else:
        possible_earlinesses = trans_starttimes - t
        possible_earlinesses = possible_earlinesses[possible_earlinesses >= 0]
        if len(possible_earlinesses) == 0: # all transitions are past time t
            return np.random.uniform(min_earliness_notrans, max_earliness)
        else: # take the closest transition happening after t
            return np.min(possible_earlinesses)

def keep_in_sample(cpstat, it, t, trans_starttimes):
    '''
    Says if this timestep for this composition must be kept in the training+evaluation sample
    '''
    if (cpstat.frac_pixdiff_inst_vs_swref.val[it] < trans_param[0] and # only keep times at which the composition is relatively stable 
        (cpstat.n_transitions == 0 or np.all(np.logical_or(t < trans_starttimes, it >= cpstat.transition_tinds[:, 3]))) and # exclude times with transition periods (can be redundant with line above)
        t >= cpstat.sw_width_sec + cpstat.tmin and # larger times than the sliding window and watchrange widths, and only when the composition is active (tmin)
        t >= watchrange + cpstat.tmin and
        t <= var.TIME_WHITEOUT and # exclude white-only period
        cpstat.area_vst.val[it] > 0 and # non-zero active area
        # t must be in a timerange where the border_path is relatively stable, and at least [watchrange] later than the start of this timerange 
        np.any(np.logical_and(t >= cpstat.stable_borders_timeranges[:, 0] + watchrange, t <= cpstat.stable_borders_timeranges[:, 1])) and
        it < cpstat.n_t_bins - 1 # exclude the last time interval, that has a different size
        ):

        # require a 3h stable region within the watchrange (so that it it the transition moment that is researched by the algo, and not the pre-transition stable period)
        len_stableregion = math.ceil(3.*3600 / (cpstat.t_lims[1] - cpstat.t_lims[0])) # maybe should use trans_param[2] (in seconds)...
        itmin = max(it - len_watchrange, np.argmax(cpstat.t_lims >= cpstat.tmin))
        stable_cond = cpstat.frac_pixdiff_inst_vs_swref.val[itmin:(it+1)] < trans_param[1]
        seq_pretrans = tran.limits_sequence_of_true(stable_cond, len_stableregion)
        return len(seq_pretrans) > 0
    
    else:
        return False


ncompmax = 12795
nevents_max = 3500000 # hard-coded (conservative) !!
inputvals = np.full((nevents_max, n_trainingvars), -1, dtype=np.float32)
outputval = np.full((nevents_max), -1, dtype=np.float32)

varnames = []
eventtime = np.full((nevents_max), -1, dtype=np.float64)
id_idx = np.full((nevents_max), -1, dtype=np.int16)
#previous_tstep_idx = np.full((nevents_max), 1e9, dtype=np.int16)
id_dict = np.full(ncompmax, '')
ntimes=0

i_event = -1
n_keptcomps = 0

period = 1000
for p in range(0, math.ceil(ncompmax/period)):
    
    # grab canvas_part_statistics object
    with open(file_path, 'rb') as f:
        cpstats = pickle.load(f)[p*period : min(ncompmax, (p+1)*period)]
        f.close()
        del f
    gc.collect()

    for icps_tmp, cps in enumerate(cpstats):
        if cps.area < 50: # exclude very small compositions # change this to 100 ?
            continue
        n_keptcomps += 1
        icps = icps_tmp + p*period
        print('cpstat #', icps, ' id ',cps.id)
        id_dict[icps] = cps.id

        cps.fill_timeseries_info()
        # all variables from this cpstat
        allvars = variables_from_cpstat(cps)
        trans_starttimes = tran.transition_start_time_simple(cps)

        # enter names for all training variables
        if icps == 0: 
            for (coarse, v) in zip(allvars[2], allvars[0]):
                timeidx = watch_timeidx_coarse if coarse else watch_timeidx
                n_times = n_traintimes_coarse if coarse else n_traintimes
                for i in range(0, n_times):
                    varnames.append(v.label + '_t' + 
                                    ((str(timeidx[i]) + str(timeidx[i+1]-1)) if (i < n_times-1) else '-0-0'))

        #prev_it = -2
        for it, t in enumerate(cps.t_lims):
            ntimes += 1
            if keep_in_sample(cps, it, t, trans_starttimes):
                i_event += 1

                vars_thistime = []
                for coarse, ratio_to_av, v in zip(allvars[2], allvars[1], allvars[0]):
                    vals = get_vals_from_var(v.ratio_to_sw_mean if ratio_to_av else v.val, it, coarse)
                    vars_thistime.extend(vals)
                # input variables
                inputvals[i_event] = vars_thistime

                # earliness output
                outputval[i_event] = get_earliness(cps, t, trans_starttimes)

                # time of this recorded event
                eventtime[i_event] = t
                #print(icps, it, t, outputval[-1], cps.frac_pixdiff_inst_vs_swref.val[it], cps.frac_pixdiff_inst_vs_swref_forwardlook.val[it], inputvals[-1][0:7] )
                id_idx[i_event] = icps

                ## index of the previous recorded event if it is only one timestep away
                #if prev_it == it-1:
                #    previous_tstep_idx[i_event] = i_event - 1
                #prev_it = it


print('# events total = ', i_event, 'from',n_keptcomps,'compositions')

varnames = np.array(varnames)
inputvals = inputvals[0:i_event+1]
outputval = outputval[0:i_event+1]
eventtime = eventtime[0:i_event+1]
id_idx = id_idx[0:i_event+1]
#previous_tstep_idx = previous_tstep_idx[0:i_event+1]

#for i in range(0, len(inputvals)):
#    print(varnames)
#    print(i, eventtime[i], inputvals[i], outputval[i])

print(inputvals.shape)
#print(inputvals)
print(outputval.shape)
#print(outputval)

file_path = os.path.join(var.DATA_PATH, 'training_data_'+str(n_trainingvars)+'variables.pickle')
with open(file_path, 'wb') as handle:
    pickle.dump([inputvals, outputval, varnames, eventtime, id_idx, id_dict, coarse_timerange, n_traintimes, n_traintimes_coarse],#previous_tstep_idx
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)

