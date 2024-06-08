import numpy as np
import pickle
import os, sys
from rplacem import var as var
import rplacem.transitions as tran
import math
import gc
import matplotlib.pyplot as plt
from rplacem.canvas_part import compare_border_paths, AtlasInfo

def variables_from_cpstat(cps):
    '''
    Array of cpstat time series variables that are kept for training/evaluation.
    The first boolean array says if the ratio of the 
    variable to its average on the preceding sliding window is taken.
    The second says if the timeranges on which averages are performed are coarse
    The third says if the kendall tau of the variable must be taken
    '''

    vars_full = [(cps.frac_pixdiff_inst_vs_swref,       0, 0, 0),
                 (cps.frac_pixdiff_inst_vs_stable_norm, 0, 0, 0),
                 (cps.frac_attack_changes,              0, 0, 0),
                 (cps.n_changes_norm,                   0, 1, 0),
                 (cps.instability_norm[0],              0, 0, 0),
                 (cps.instability_norm[3],              0, 0, 0),#
                 (cps.variance2,                        0, 0, 0),
                 (cps.variance_multinom,                0, 0, 0),#n
                 (cps.variance_subdom,                  0, 0, 0),#n
                 (cps.variance_from_frac_pixdiff_inst,  0, 0, 0),#n
                 (cps.runnerup_timeratio[0],            0, 0, 0),#
                 (cps.runnerup_timeratio[3],            0, 0, 0),
                 (cps.n_used_colors[0],                 0, 0, 0),
                 (cps.n_used_colors[3],                 0, 1, 0),
                 (cps.autocorr_bycase,                  1, 0, 0),#  # the difference is taken instead of the ratio for autocorrelation
                 (cps.autocorr_bycase_norm,             1, 0, 0),# # the difference is taken instead of the ratio for autocorrelation
                 (cps.autocorr_multinom,                0, 0, 0),#n
                 (cps.autocorr_subdom,                  0, 0, 0),#n
                 (cps.autocorr_dissimil,                0, 0, 0),#n
                 (cps.cumul_attack_timefrac,            0, 1, 0),#
                 (cps.returntime[0],                    0, 1, 0),
                 (cps.returntime[3],                    0, 1, 0),
                 (cps.returnrate,                       0, 0, 0),#n
                 (cps.n_users_sw_norm,                  0, 1, 0),
                 (cps.changes_per_user_sw,              0, 1, 0),
                 (cps.frac_users_new_vs_sw,             0, 0, 0),
                 (cps.frac_redundant_color_changes,     0, 0, 0),
                 (cps.entropy,                          1, 0, 0),
                 (cps.fractal_dim_weighted,             1, 0, 0),
                 (cps.variance2,                        0, 0, 1),#
                 (cps.returntime[0],                    0, 0, 1),#
                 (cps.instability_norm[0],              0, 0, 1),#
                 (cps.autocorr_bycase,                  0, 0, 1),#
                 (cps.autocorr_bycase_norm,             0, 0, 1),#
                ]
    
    vars = np.array([v[0] for v in vars_full])
    take_ratio_to_average = np.array([v[1] for v in vars_full])
    coarse_timeranges = np.array([v[2] for v in vars_full])
    take_kendall = np.array([v[3] for v in vars_full])

    return(vars, take_ratio_to_average, coarse_timeranges, take_kendall)

def set_kendalls_and_ratio(vars, take_ratio, take_kendall):
    for i in np.arange(vars.shape[0]):
        if take_ratio[i]:
            vars[i].set_ratio_to_sw_average()
        if take_kendall[i]:
            vars[i].sw_width_ews = int(3600/cps.t_interval) * 1 # 1h
            vars[i].set_kendall_tau(rerun=True)
    


def get_vals_from_var(v, tidx, coarse=False):
    '''
    Return values of the variable at all watch timeranges for index [tidx - watch_timeidx]
    Variables are averaged over values when watch time indices have a separation > 1
    '''
    n_times = n_traintimes_coarse if coarse else n_traintimes
    vals = np.empty(n_times, dtype=np.float64)
    v = np.nan_to_num(v) ####### TODO To be removed when using updated cpart stats

    for i in range(0, n_times):
        inds = tidx + (watch_timeindrange_coarse if coarse else watch_timeindrange)[i]
        vals[i] = np.mean(v[inds])
    return vals

def get_earliness(cpstat, t, trans_starttimes):
    '''
    Return the earliness of a potential signal at time t compared to the transitions present in cpstat.
    All earliness values above 12h (and when there is no transition in the composition) are given a value of exactly 12h
    '''
    earliness_notrans = 12*3600 # high earliness value given to events in no-transition compositions or events with a transition and high earliness

    if cpstat.n_transitions == 0: # no transitions
        return earliness_notrans
    else:
        possible_earlinesses = trans_starttimes - t
        possible_earlinesses = possible_earlinesses[possible_earlinesses >= 0]
        if len(possible_earlinesses) == 0: # all transitions are before time t
            return earliness_notrans 
        else: # take the closest transition happening after t
            return min(np.min(possible_earlinesses), earliness_notrans)

def keep_in_sample(cpstat, it, t, trans_starttimes, reject_times):
    '''
    Says if this timestep for this composition must be kept in the training+evaluation sample
    '''
    trans_param = cpstat0.transition_param
    #print()
    #print(cpstat.area >= 100 ) # exclude very small compositions
    #print(cpstat.frac_pixdiff_inst_vs_swref.val[it] < trans_param[0] or 
    #     cpstat.frac_pixdiff_inst_vs_swref.ratio_to_sw_mean[it] < trans_param[1])
    #print(cpstat.n_transitions == 0 or np.all(np.logical_or(t < trans_starttimes, 
    #                                                       t >= np.maximum(cpstat.transition_times[:, 1], 
    #                                                                       trans_starttimes + cpstat.sw_width_sec)))) 
    #print(t <= var.TIME_GREYOUT)
    #print(cpstat.area_vst.val[it] > 0) # non-zero active area
    #print(cpstat.stable_borders_timeranges)
    #print(np.any(np.logical_and(t >= cpstat.stable_borders_timeranges[:, 0] + max(watchrange, cpstat.sw_width_sec), 
    #                          t <= cpstat.stable_borders_timeranges[:, 1])))
    #print(it < cpstat.n_t_bins - 1)
    return (cpstat.area >= 100 and # exclude very small compositions
            (cpstat.frac_pixdiff_inst_vs_swref.val[it] < trans_param[0] or 
             cpstat.frac_pixdiff_inst_vs_swref.ratio_to_sw_mean[it] < trans_param[1]) and # excludes times where the composition has transition-like values for frac_pixdiff
            (cpstat.n_transitions == 0 or np.all(np.logical_or(t < trans_starttimes, 
                                                               t >= np.maximum(cpstat.transition_times[:, 1], 
                                                                               trans_starttimes + cpstat.sw_width_sec)))) and # exclude times within transition periods, and within sw_width of the start of the transition
            t <= var.TIME_GREYOUT and # exclude white-only period
            cpstat.area_vst.val[it] > 0 and # non-zero active area
            # t must be in a timerange where the border_path is relatively stable, and at least [watchrange] and [sw_width] later than the start of this timerange 
            np.any(np.logical_and(t >= cpstat.stable_borders_timeranges[:, 0] + max(watchrange, cpstat.sw_width_sec), 
                               t <= cpstat.stable_borders_timeranges[:, 1])) and
            it < cpstat.n_t_bins - 1 and # exclude the last time interval, that has a different size
            # exclude time that overlap in space and time with another composition
            np.all((t > reject_times[:,1]) | (t < reject_times[:,0]))
            # must be below absolute transition threshold (not the relative one) for a duration > sliding_window_width right before this time
            #and np.all(cpstat.frac_pixdiff_inst_vs_swref.val[(it-cpstat.sw_width) : it] < trans_param[0])
            )

file_path = os.path.join(var.DATA_PATH, 'canvas_composition_statistics_all.pickle') 

def get_all_AtlasInfo():
    path = os.path.join(var.DATA_PATH, 'canvas_compositions_statistics_all.pickle') 
    infos = []
    with open(path, 'rb') as f:
        canvas_parts = pickle.load(f)
        for cp in canvas_parts:
            infos.append(cp.info)
    gc.collect()
    return infos

def duplicate_place_and_time():
    with open(file_path, 'rb') as f:
        cpstats = pickle.load(f)

        # get border_path associated to each cpstat
        infos_list = []
        for icps, cps in enumerate(cpstats):
            if not hasattr(cps, 'info'): # this won't be needed with the new CanvasPartStatistics
                if icps == 0:
                    infos = get_all_AtlasInfo()
                found = False
                for inf in infos:
                    if inf.id == cps.id:
                        infos_list.append(inf)
                        found = True
                        break
                if not found:
                    infos_list.append(AtlasInfo()) # dummy here, these compositions won't reject any duplicates
            else:
                infos_list.append(cps.info)
        
        rejected_times_border_overlaps = np.zeros(len(cpstats), dtype=object)
        # first loop on each border_path of each cpstats 
        for icps in range(len(cpstats)):
            #print(icps)
            borders = infos_list[icps]
            for bpath, btimes in zip(borders.border_path, borders.border_path_times):
                if bpath.size == 0: # not needed when we will run with the new cpstats
                    continue
                # second loop on each border_path of each cpstats 
                for icps2 in range(len(cpstats)):
                    if icps2 <= icps: # only triangular 2D test
                        continue
                    borders2 = infos_list[icps2]
                    for bpath2, btimes2 in zip(borders2.border_path, borders2.border_path_times):
                        if bpath2.size == 0: # not needed when we will run with the new cpstats
                            continue

                        # now we have a specific border_path for both cpstats, let's compare them, first their time overlap
                        times_overlap = [max(btimes[0], btimes2[0]), min(btimes[1], btimes2[1])]
                        if times_overlap[0] < times_overlap[1]:
                            # then their physical overlap (fraction of overlapping pixels)
                            borders_overlap = compare_border_paths(bpath, bpath2)
                            if borders_overlap > 0.9:
                                # for these overlapping compositions, keep the one with the earliest start time
                                icps_reject = icps2 if (btimes[0] < btimes2[0] or (np.isclose(btimes[0], btimes2[0]) and btimes[1] >= btimes2[1])) else icps
                                if rejected_times_border_overlaps[icps_reject] == 0:
                                    rejected_times_border_overlaps[icps_reject] = [times_overlap]
                                else:
                                    rejected_times_border_overlaps[icps_reject].append(times_overlap)
                                print('        ',icps,icps2,icps_reject,borders_overlap,times_overlap)
                                #print('        ',bpath,bpath2)
                                print('        ',btimes,btimes2)
                                print(borders.id, borders2.id)

        return rejected_times_border_overlaps    

# rejected times due to overlapping compositions
run_rejecttimes = False
file_path_rejecttimes = os.path.join(var.DATA_PATH, 'reject_times_from_overlap.pickle')
if run_rejecttimes:
    reject_times_all = duplicate_place_and_time()
    with open(file_path_rejecttimes, 'wb') as f:
        pickle.dump(reject_times_all, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('# of overlaps =', np.count_nonzero(reject_times_all))
else:
    with open(file_path_rejecttimes, 'rb') as f:
        reject_times_all = pickle.load(f)


# grab canvas_part_statistics object
with open(file_path, 'rb') as f:
    cpstat0 = pickle.load(f)[0]
    f.close()
    del f
gc.collect()

'''
areas = []
with open(file_path, 'rb') as f:
    cpstats = pickle.load(f)
    areas = [cps.area for cps in cpstats]
    f.close()
plt.figure()
plt.hist(areas, bins=np.logspace(np.log10(1),np.log10(1e5), 100), range=(1,1e5))
plt.xlabel('composition size in pixels')
plt.ylabel('counts')
plt.xscale('log')   
plt.xlim([1,1e5])
plt.savefig(os.path.join(var.FIGS_PATH, 'size_of_all_compositions.png'))
print(np.count_nonzero(np.array(areas) >= 50))
print(np.count_nonzero(np.array(areas) >= 0))
'''

# Define times that are kept before each event used in 
tstep = cpstat0.t_interval
watchrange_min = 7*3600 # minimal time range kept for single time series instances in training

watch_timeidx_diff = np.array([1, 1, 2, 2, 3, 4, 5, 7, 15, 15, 15, 15]) #np.array([1, 1, 2, 5, 13, 17, 17, 17]) # time indices kept, preceding the time of an event 
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
watch_timeidx_diff_coarse = np.array([1, 3, 4, 6, 11, 15, 15, 15, 15])#np.array([1, 8, 13, 17, 17, 17]) # time indices kept. For smoothened variables
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


cpstatvars = variables_from_cpstat(cpstat0)
coarse_timerange = cpstatvars[2]
kendall_tau = cpstatvars[3]
n_cpstatvars = np.count_nonzero(coarse_timerange == 0)
n_cpstatvars_coarse = np.count_nonzero(coarse_timerange == 1)
n_trainingvars = n_cpstatvars * n_traintimes + n_cpstatvars_coarse * n_traintimes_coarse

ncompmax = 14222 if var.year == 2022 else 6720
nevents_max = 4000000 # hard-coded (conservative) !!
inputvals = np.full((nevents_max, n_trainingvars), -1, dtype=np.float32)
outputval = np.full((nevents_max), -1, dtype=np.float32)

varnames = []
eventtime = np.full((nevents_max), -1, dtype=np.float64)
id_idx = np.full((nevents_max), -1, dtype=np.int16)
id_dict = np.full(ncompmax, '', dtype='object')

i_event = -1
n_keptcomps = 0

period = 1000
istart = 0 
for p in range(istart, math.ceil(ncompmax/period)):
    
    # grab canvas_part_statistics object
    with open(file_path, 'rb') as f:
        cpstats = pickle.load(f)[p*period : min(ncompmax, (p+1)*period)]
        f.close()
        del f
    gc.collect()

    for icps_tmp, cps in enumerate(cpstats):
        icps = icps_tmp + p*period
        print('cpstat #', icps, ' id ',cps.id, ' area ', cps.area)
        #if icps<252:
        #    continue
        id_dict[icps] = cps.id
        cps.stable_borders_timeranges[:, 0] = np.maximum(cps.stable_borders_timeranges[:, 0], cps.tmin)

        cps.fill_timeseries_info()
        # all variables from this cpstat
        allvars = variables_from_cpstat(cps)
        trans_starttimes = tran.transition_start_time_simple(cps)
        reject_times = reject_times_all[icps]
        if reject_times == 0:
            reject_times = [[1e8, -1]]
        elif len(reject_times) == 0:
            reject_times = [[1e8, -1]]
        reject_times = np.array(reject_times)

        # enter names for all training variables
        if icps == istart: 
            for (coarse, v) in zip(allvars[2], allvars[0]):
                timeidx = watch_timeidx_coarse if coarse else watch_timeidx
                n_times = n_traintimes_coarse if coarse else n_traintimes
                for i in range(0, n_times):
                    varnames.append(v.label + '_t' + 
                                    ((str(timeidx[i]) + str(timeidx[i+1]-1)) if (i < n_times-1) else '-0-0'))


        i_event_thiscomp = 0
        for it, t in enumerate(cps.t_lims):
            if keep_in_sample(cps, it, t, trans_starttimes, reject_times):
                i_event += 1
                i_event_thiscomp += 1

                if i_event_thiscomp == 1:
                    set_kendalls_and_ratio(allvars[0], allvars[1], allvars[3])

                vars_thistime = []
                for kendall, coarse, ratio_to_av, v in zip(allvars[3], allvars[2], allvars[1], allvars[0]):
                    vals = get_vals_from_var(v.kendall_tau if kendall else v.ratio_to_sw_mean if ratio_to_av else v.val, it, coarse)
                    vars_thistime.extend(vals)
                # input variables
                inputvals[i_event] = vars_thistime

                # earliness output
                outputval[i_event] = get_earliness(cps, t, trans_starttimes)

                # time of this recorded event
                eventtime[i_event] = t
                id_idx[i_event] = icps
                
        if i_event_thiscomp > 0:
            n_keptcomps += 1    
        print("Added",i_event_thiscomp,"timesteps from this compo. Now there are ",i_event,"events")


print('# events total = ', i_event, 'from',n_keptcomps,'compositions')

varnames = np.array(varnames)
inputvals = inputvals[0:i_event+1]
outputval = outputval[0:i_event+1]
eventtime = eventtime[0:i_event+1]
id_idx = id_idx[0:i_event+1]

#for i in range(0, len(inputvals)):
#    print(varnames)
#    print(i, eventtime[i], inputvals[i], outputval[i])

print(inputvals.shape)
#print(inputvals)
print(outputval.shape)
#print(outputval)

file_path = os.path.join(var.DATA_PATH, 'training_data_'+str(n_trainingvars)+'variables.pickle')
with open(file_path, 'wb') as handle:
    pickle.dump([inputvals, outputval, varnames, eventtime, id_idx, id_dict, coarse_timerange, kendall_tau, n_traintimes, n_traintimes_coarse],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)

