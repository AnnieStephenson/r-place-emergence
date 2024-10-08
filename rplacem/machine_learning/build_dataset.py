import numpy as np
import pickle
import os, sys
from rplacem import var as var
import rplacem.transitions as tran
import rplacem.entropy as entropy
import math
import gc
import matplotlib.pyplot as plt
from rplacem.canvas_part import compare_border_paths, AtlasInfo, avoid_location_jump

param_num = 0 # 0 is nominal result, 1 to 6 are sensitivity analysis
param_str = var.param_str_fun(param_num)
file_path = os.path.join(var.DATA_PATH, 'cpart_stats_'+param_str+'_'+str(var.year)+'.pkl') 

def border_corner_center(xmin,xmax,ymin,ymax):
    # determines if the composition with the input x,y limits 
    # (note that they are the limits of the smallest rectangle containing all border paths of the composition)
    # is in the border, corner, or center of the available canvas space. 

    cond_border = np.zeros((var.N_ENLARGE))
    cond_corner = np.zeros((var.N_ENLARGE))
    cond_center = np.zeros((var.N_ENLARGE))
    margin = 3
    for e in range(var.N_ENLARGE):
        cond_border[e] = ((xmin < var.CANVAS_MINMAX[e, 0, 0] + margin) |
                            (ymin < var.CANVAS_MINMAX[e, 1, 0] + margin) |
                            (xmax > var.CANVAS_MINMAX[e, 0, 1] - margin) |
                            (ymax > var.CANVAS_MINMAX[e, 1, 1] - margin) ) 
        cond_corner[e] = (
                            ((xmin < var.CANVAS_MINMAX[e, 0, 0] + margin) &
                            (ymin < var.CANVAS_MINMAX[e, 1, 0] + margin)) |
                            ((xmax > var.CANVAS_MINMAX[e, 0, 1] - margin) &
                            (ymax > var.CANVAS_MINMAX[e, 1, 1] - margin) ) |
                            ((xmin < var.CANVAS_MINMAX[e, 0, 0] + margin) &
                            (ymax > var.CANVAS_MINMAX[e, 1, 1] - margin) ) |
                            ((xmax > var.CANVAS_MINMAX[e, 0, 1] - margin) &
                            (ymin < var.CANVAS_MINMAX[e, 1, 0] + margin) )
                         )
        x_center = (var.CANVAS_MINMAX[e, 0, 0] + var.CANVAS_MINMAX[e, 0, 1]) / 2
        y_center = (var.CANVAS_MINMAX[e, 1, 0] + var.CANVAS_MINMAX[e, 1, 1]) / 2
        cond_center[e] = ((xmin < x_center + margin) & 
                            (xmax > x_center - margin) &
                            (ymin < y_center + margin) & 
                            (ymax > y_center - margin) )
    return cond_border, cond_corner, cond_center

def variables_from_cpstat(cps):
    '''
    Array of cpstat time series variables that are kept for training/evaluation.
    The first boolean array says if the ratio of the 
    variable to its average on the preceding sliding window is taken.
    The second says if the timeranges on which averages are performed are coarse
    The third says if the kendall tau of the variable must be taken
    '''

    vars_full = [(cps.frac_pixdiff_inst_vs_swref,       0, 0, 0),#0
                 (cps.frac_pixdiff_inst_vs_stable_norm, 0, 0, 0),#1
                 (cps.frac_attack_changes,              0, 0, 0),#2
                 (cps.n_changes_norm,                   0, 1, 0),#3
                 (cps.instability_norm[0],              0, 0, 0),#4
                 (cps.instability_norm[3],              0, 0, 0),#5
                 (cps.variance2,                        0, 0, 0),#6
                 (cps.variance_multinom,                0, 0, 0),#7
                 (cps.variance_subdom,                  0, 0, 0),#8
                 (cps.variance_from_frac_pixdiff_inst,  0, 1, 0),#9
                 (cps.runnerup_timeratio[0],            0, 0, 0),#10
                 (cps.runnerup_timeratio[3],            0, 0, 0),#11
                 (cps.n_used_colors[0],                 0, 0, 0),#12
                 (cps.n_used_colors[3],                 0, 1, 0),#13
                 (cps.autocorr_bycase,                  1, 0, 0),#14# the difference is taken instead of the ratio for autocorrelation
                 (cps.autocorr_bycase_norm,             1, 0, 0),#15#   # the difference is taken instead of the ratio for autocorrelation
                 (cps.autocorr_multinom,                0, 0, 0),#16
                 (cps.autocorr_subdom,                  0, 0, 0),#17
                 (cps.autocorr_dissimil,                0, 0, 0),#18
                 (cps.cumul_attack_timefrac,            0, 1, 0),#19
                 (cps.returntime[0],                    0, 1, 0),#20
                 (cps.returntime[3],                    0, 1, 0),#21
                 (cps.returnrate,                       0, 0, 0),#22
                 (cps.n_users_sw_norm,                  0, 1, 0),#23
                 (cps.changes_per_user_sw,              0, 1, 0),#24
                 (cps.frac_users_new_vs_sw,             0, 0, 0),#25
                 (cps.frac_redundant_color_changes,     0, 0, 0),#26
                 (cps.entropy,                          1, 0, 0),#27
                 (cps.fractal_dim_weighted,             1, 0, 0),#28
                 (cps.variance_multinom,                0, 0, 1),#
                 (cps.returntime[0],                    0, 0, 1),#
                 (cps.instability_norm[0],              0, 0, 1),#
                 (cps.autocorr_bycase,                  0, 0, 1),#
                 (cps.autocorr_bycase_norm,             0, 0, 1),#
                 (cps.variance_from_frac_pixdiff_inst,  0, 0, 1),#
                ]
    
    vars = np.array([v[0] for v in vars_full])
    take_ratio_to_average = np.array([v[1] for v in vars_full])
    coarse_timeranges = np.array([v[2] for v in vars_full])
    take_kendall = np.array([v[3] for v in vars_full])

    return(vars, take_ratio_to_average, coarse_timeranges, take_kendall)

def additional_vars_from_cpstat(cps, t, it, timerange, special_pos):
    entropy_sw = entropy.normalize_entropy(cps.entropy.val[it] / cps.entropy.ratio_to_sw_mean[it], 
                                      cps.area, t, minmax_entropy=minmax_entropy)
    quadrant_time = np.searchsorted(var.TIME_ENLARGE, t) - 1
    pos_importance = special_pos[0][quadrant_time] + special_pos[1][quadrant_time] + special_pos[2][quadrant_time]
    return [np.log10(cps.area), t - timerange[0], cps.quadrant, entropy_sw, pos_importance ]

def additional_vars_names():
    return ['log(area)', 'age', 'canvas_quadrant', 'entropy_sw', 'border_corner_center']

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
        return - earliness_notrans
    else:
        possible_earlinesses = trans_starttimes - t
        possible_earlinesses = possible_earlinesses[possible_earlinesses >= 0]
        if len(possible_earlinesses) == 0: # all transitions are before time t
            return - earliness_notrans 
        else: # take the closest transition happening after t
            return - min(np.min(possible_earlinesses), earliness_notrans)

def keep_in_sample(cpstat, it, t, trans_starttimes, reject_times, atlas_timerange):
    '''
    Says if this timestep for this composition must be kept in the training+evaluation sample
    '''
    trans_param = cpstat0.transition_param

    return ((cpstat.area >= 100 and # exclude very small compositions
            # exclude times after official end of composition in the atlas. Leave some margin for transition at the end of compositions
            t < atlas_timerange[1] + (3600 if np.any(trans_starttimes) > atlas_timerange[1] else 0) and 
            # exclude times before official beginning of composition in the atlas + sw_width
            t > atlas_timerange[0] + cpstat.sw_width_sec and

            # excludes times where the composition has transition-like values for frac_pixdiff
            (cpstat.frac_pixdiff_inst_vs_swref.val[it] < trans_param[0] or 
             cpstat.frac_pixdiff_inst_vs_swref.ratio_to_sw_mean[it] < trans_param[1]) and 
             # exclude times within sw_width of the start of a transition
            (cpstat.n_transitions == 0 or 
             np.all(np.logical_or(t < trans_starttimes, t >= trans_starttimes + cpstat.sw_width_sec))) and 

            # t must be in a timerange where the border_path is relatively stable, and at least [watchrange + sw_width] later than the start of this timerange 
            # stable_borders_timeranges lower time limit cannot be < tmin_quadrant. 
            # the earliest timerange of stable_borders_timeranges is artificially lower with addtime_before, so this criterium is irrelevant for the first timerange borderpath, except close to tmin_quadrant
            np.any(np.logical_and(t >= cpstat.stable_borders_timeranges[:, 0] + watchrange + cpstat.sw_width_sec, 
                                  t <= cpstat.stable_borders_timeranges[:, 1])) and
            # exclude time that overlap in space and time with another composition
            np.all((t > reject_times[:,1]) | (t < reject_times[:,0])) and 

            t <= var.TIME_GREYOUT and # exclude white-only period
            cpstat.area_vst.val[it] > 0 and # non-zero active area
            it < cpstat.n_t_bins - 1 # exclude the last time interval, that has a different size
            # must be below absolute transition threshold (not the relative one) for a duration > sliding_window_width right before this time
            #and np.all(cpstat.frac_pixdiff_inst_vs_swref.val[(it-cpstat.sw_width) : it] < trans_param[0])
            ),

            # keep info about more secure time margin after atlas start or previous transition
            (t > atlas_timerange[0] + watchrange + cpstat.sw_width_sec),
            (cpstat.n_transitions == 0 or 
             np.all(np.logical_or(t < trans_starttimes, t >= trans_starttimes + watchrange + cpstat.sw_width_sec)))
            )

def get_all_AtlasInfo():
    infos = []
    with open(file_path, 'rb') as f:
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
            infos_list.append(cps.info)
        
        rejected_times_border_overlaps = np.zeros(len(cpstats), dtype=object)
        ids = np.zeros(len(cpstats), dtype=object)
        # first loop on each border_path of each cpstats 
        for icps in range(len(cpstats)):
            #print(icps)
            borders = infos_list[icps]
            ids[icps] = borders.id
            for bpath, btimes in zip(borders.border_path, borders.border_path_times):
                #if bpath.size == 0: # not needed when we will run with the new cpstats
                #    continue
                # second loop on each border_path of each cpstats 
                for icps2 in range(len(cpstats)):
                    if icps2 <= icps: # only triangular 2D test
                        continue
                    borders2 = infos_list[icps2]
                    for bpath2, btimes2 in zip(borders2.border_path, borders2.border_path_times):
                        #if bpath2.size == 0: # not needed when we will run with the new cpstats
                        #    continue

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

        return rejected_times_border_overlaps, ids   

# rejected times due to overlapping compositions in the initial atlas (different entries with almost (>90%) the same border_path)
run_rejecttimes = False
file_path_rejecttimes = os.path.join(var.DATA_PATH, 'reject_times_from_overlapping_comps_'+param_str+'.pickle')
if run_rejecttimes:
    if param_num==0:
        reject_times_all, compo_ids = duplicate_place_and_time()
    
    else:
        file_path_rejecttimes_nom = os.path.join(var.DATA_PATH, 'reject_times_from_overlapping_comps_'+var.param_str_fun(0)+'.pickle')
        with open(file_path_rejecttimes_nom, 'rb') as f:
            reject_times_all_nom, compo_ids = pickle.load(f)

        # this is to recover the order of the compositions from the param_num==0 nominal run
        with open(file_path, 'rb') as f:
            cpstats = pickle.load(f)
        reject_times_all = np.copy(reject_times_all_nom)
        id_list = []

        for icps, cps in enumerate(cpstats):
            id = cps.info.id
            id_list.append(id)
            compnum_nom = np.argmax(compo_ids == id) # index of that composition id in the nominal stored result
            reject_times_all[icps] = reject_times_all_nom[compnum_nom]

    with open(file_path_rejecttimes, 'wb') as f:
        pickle.dump([reject_times_all, None if param_num>0 else compo_ids], f, protocol=pickle.HIGHEST_PROTOCOL)
    print('# of overlaps =', np.count_nonzero(reject_times_all))    
    sys.exit()

else:
    with open(file_path_rejecttimes, 'rb') as f:
        reject_times_all, _ = pickle.load(f)

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
n_trainingvars = n_cpstatvars * n_traintimes + n_cpstatvars_coarse * n_traintimes_coarse + 5

ncompmax = 14300 if var.year == 2022 else 6950
nevents_max = 3600000 # hard-coded (conservative) !!
inputvals = np.full((nevents_max, n_trainingvars), -1, dtype=np.float32)
safetimemargin = np.full((nevents_max, 2), -1, dtype=np.float32)
outputval = np.full((nevents_max), -1, dtype=np.float32)

varnames = []
eventtime = np.full((nevents_max), -1, dtype=np.float64)
id_idx = np.full((nevents_max), -1, dtype=np.int16)
id_dict = np.full(ncompmax, '', dtype='object')
minmax_entropy = entropy.calc_min_max_entropy()

i_event = -1
n_keptcomps = 0

period = 500
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
        #if icps<3443:
        #    continue
        print('cpstat #', icps, ' id ',cps.id, ' area ', cps.area)
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
            varnames.extend(additional_vars_names())

        # time range from the original borderpath of the atlas (though separated when disjoint in space or time)
        atlas_timerange = [ cps.info.border_path_times_orig_disjoint[0][0], 
                            cps.info.border_path_times_orig_disjoint[-1][1] ]

        xmin, xmax = np.min(cps.info.border_path[:,:, 0]), np.max(cps.info.border_path[:,:, 0])
        ymin, ymax = np.min(cps.info.border_path[:,:, 1]), np.max(cps.info.border_path[:,:, 1])        
        special_pos = border_corner_center(xmin,xmax,ymin,ymax)

        i_event_thiscomp = 0
        for it, t in enumerate(cps.t_lims):
            keep = keep_in_sample(cps, it, t, trans_starttimes, reject_times, atlas_timerange)
            if keep[0]:
                i_event += 1
                i_event_thiscomp += 1

                if i_event_thiscomp == 1:
                    set_kendalls_and_ratio(allvars[0], allvars[1], allvars[3])

                vars_thistime = []
                for kendall, coarse, ratio_to_av, v in zip(allvars[3], allvars[2], allvars[1], allvars[0]):
                    vals = get_vals_from_var(v.kendall_tau if kendall else v.ratio_to_sw_mean if ratio_to_av else v.val, it, coarse)
                    vars_thistime.extend(vals)
                vars_thistime.extend(additional_vars_from_cpstat(cps, t, it, atlas_timerange, special_pos))

                # input variables
                inputvals[i_event] = vars_thistime

                # earliness output
                outputval[i_event] = get_earliness(cps, t, trans_starttimes)

                # time of this recorded event
                eventtime[i_event] = t
                id_idx[i_event] = icps

                # flags for the safer margin after start of composition or after last transition
                safetimemargin[i_event] = [keep[1], keep[2]]
               
        if i_event_thiscomp > 0:
            n_keptcomps += 1    
            print("Added",i_event_thiscomp,"timesteps from this compo. Now there are ",i_event,"events")

print('# events total = ', i_event, 'from',n_keptcomps,'compositions')

varnames = np.array(varnames)
inputvals = inputvals[0:i_event+1]
outputval = outputval[0:i_event+1]
eventtime = eventtime[0:i_event+1]
safetimemargin = safetimemargin[0:i_event+1]
id_idx = id_idx[0:i_event+1]

print(inputvals.shape)
print(outputval.shape)

file_path_out = os.path.join(var.DATA_PATH, 'training_data_'+str(n_trainingvars)+'variables_'+param_str+'.pickle')
with open(file_path_out, 'wb') as handle:
    pickle.dump([inputvals, outputval, varnames, eventtime, id_idx, id_dict, coarse_timerange, kendall_tau, n_traintimes, n_traintimes_coarse, safetimemargin],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)

