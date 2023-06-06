import numpy as np
import pickle
import os
import rplacem.variables_rplace2022 as var
import rplacem.canvas_part_statistics as stat
import rplacem.transitions as tran

# grab canvas_part_statistics object
file_path = os.path.join(var.DATA_PATH, 'canvas_composition_statistics_0_to_200.pickle') 
with open(file_path, 'rb') as f:
    cpstats = pickle.load(f)[0:100]

# Define times that are kept before each event used in 
tstep = cpstats[0].t_interval
trans_param = cpstats[0].transition_param

watchrange_min = max(trans_param[2] + trans_param[3], 5*3600) # minimal time range kept for single time series instances in training
watch_timeidx_diff = np.array([1, 1, 1, 2, 3, 6, 9, 14, 16, 20]) #[1, 1, 1, 1, 2, 3, 4, 6, 8, 10]) # time indices kept, preceding the time of an event 
while tstep * (np.sum(watch_timeidx_diff)-1) < watchrange_min:
    watch_timeidx_diff = np.append(watch_timeidx_diff, 12)
watch_timeidx = - np.flip(np.cumsum(watch_timeidx_diff)) + 1 # starts at -max_watch_index and ends at 0
print(watch_timeidx)
n_traintimes = watch_timeidx.shape[0] # number of values in time kept for each variable of each event
watch_timeindrange = np.empty(n_traintimes, dtype=object) # 1d array of 1d lists of indices that will be averaged over
for i in range(0, n_traintimes):
    watch_timeindrange[i] = np.arange(watch_timeidx[i], ( 1 if (i == n_traintimes-1) else watch_timeidx[i+1] ))
watchrange = tstep * ( - watch_timeidx[0] )
print(watchrange)

def variables_from_cpstat(cps):
    '''
    Array of cpstat time series variables that are kept for training/evaluation.
    The first boolean array says if the ratio of the 
    variable to its average on the preceding sliding window is taken.
    '''
    cps.entropy.set_ratio_to_sw_average()
    cps.fractal_dim_weighted.set_ratio_to_sw_average()

    return (np.array([0,0,0,0,0,0,0,0,1,1], dtype=np.bool),
            np.array([cps.frac_pixdiff_inst_vs_swref,
                     cps.frac_pixdiff_inst_vs_stable_norm,
                     cps.n_changes_norm,
                     cps.frac_attack_changes,
                     cps.returntime_median_overln2,
                     cps.returntime_mean,
                     cps.frac_redundant_color_changes,
                     cps.instability_norm,
                     cps.entropy,
                     cps.fractal_dim_weighted,
                     #cps.cumul_attack_timefrac,
                    ])
            )

n_cpstatvars = variables_from_cpstat(cpstats[0])[1].shape[0]
n_trainingvars = n_cpstatvars * n_traintimes

def get_vals_from_var(v, tidx):
    '''
    Return values of the variable at all preceding watch time indices [tidx - watch_timeidx]
    Variables are averaged over values when watch time indices have a separation > 1
    '''
    vals = np.empty(n_traintimes, dtype=np.float64)
    for i in range(0, n_traintimes):
        inds = tidx + watch_timeindrange[i]
        vals[i] = np.mean(v[inds])
    return vals

def get_earlyness(cpstat, t, trans_starttimes):
    '''
    Return the earlyness of a potential signal at time t compared to the transitions present in cpstat
    '''
    max_earlyness = var.TIME_WHITEONLY - max(cpstat.sw_width_sec, watchrange) # max range of earlyness
    if cpstat.n_transitions == 0: # no transitions
        return max_earlyness
    else:
        possible_earlynesses = trans_starttimes - t
        possible_earlynesses = possible_earlynesses[possible_earlynesses >= 0]
        if len(possible_earlynesses) == 0: # all transitions are past time t
            return max_earlyness
        else: # take the closest transition happening after t
            return np.min(possible_earlynesses)

def keep_in_sample(cpstat, it, t, trans_starttimes):
    '''
    Says if this timestep for this composition must be kept in the training+evaluation sample
    '''
    return (cpstat.frac_pixdiff_inst_vs_swref.val[it] < trans_param[0] and # only keep times at which the composition is relatively stable 
            (cpstat.n_transitions == 0 or np.all(np.logical_or(t < trans_starttimes, it >= cpstat.transition_tinds[:, 3]))) and # exclude times with transition periods (can be redundant with line above)
            t >= cpstat.sw_width_sec and # larger times than the sliding window and watchrange widths
            t >= watchrange + cpstat.tmin and
            t < var.TIME_WHITEONLY and # exclude white-only period
            cpstat.area_vst.val[it] > 0 and # non-zero active area
            np.any(np.logical_and(t >= cpstat.stable_borders_timeranges[:, 0] + watchrange, t <= cpstat.stable_borders_timeranges[:, 1]))
            )


inputvals = []#np.zeros((ncomp, nfeat), dtype=np.float64)
outputval = []

varnames = []
eventtime = []

for icps, cps in enumerate(cpstats):
    print('cpstat #', icps, ' id ',cps.id)
    cps.fill_timeseries_info()
    # all variables from this cpstat
    allvars = variables_from_cpstat(cps)
    trans_starttimes = tran.transition_start_time_simple(cps)


    # enter names for all training variables
    if icps == 0: 
        for v in allvars[1]:
            for i in range(0, n_traintimes):
                varnames.append(v.label + '_t' + 
                                ((str(watch_timeidx[i]) + str(watch_timeidx[i+1]-1)) if (i < n_traintimes-1) else '-0-0'))

    for it, t in enumerate(cps.t_lims):
        if keep_in_sample(cps, it, t, trans_starttimes):
            vars_thistime = []
            for ratio_to_av, v in allvars[0], allvars[1]:
                vals = get_vals_from_var(v.ratio_to_sw_mean if ratio_to_av else v.val, it)
                vars_thistime.extend(vals)
            inputvals.append(vars_thistime)

            # earlyness output
            outputval.append(get_earlyness(cps, t, trans_starttimes))

            # time of this recorded event
            eventtime.append(t)
            #print(icps, it, t, outputval[-1], cps.frac_pixdiff_inst_vs_swref.val[it], cps.frac_pixdiff_inst_vs_swref_forwardlook.val[it], inputvals[-1][0:7] )


varnames = np.array(varnames)
inputvals = np.array(inputvals)
outputval = np.array(outputval)
eventtime = np.array(eventtime)
#for i in range(0, len(inputvals)):
#    print(varnames)
#    print(i, eventtime[i], inputvals[i], outputval[i])

print(inputvals.shape)
#print(inputvals)
print(outputval.shape)
#print(outputval)


file_path = os.path.join(var.DATA_PATH, 'training_data.pickle')
with open(file_path, 'wb') as handle:
    pickle.dump([inputvals, outputval],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)

