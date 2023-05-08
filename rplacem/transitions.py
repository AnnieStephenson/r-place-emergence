import numpy as np
import rplacem.canvas_part as cp
import rplacem.compute_variables as comp
import rplacem.utilities as util
import rplacem.variables_rplace2022 as var
import math
from scipy import optimize

def find_transitions(t_lims,
                     testvariable,
                     cutoff=0.2, 
                     cutoff_stable=0.05,
                     len_stableregion=2*3600,
                     distfromtrans_stableregion=3*3600,
                     sliding_win=0
                     ):
    '''
    identifies transitions in a cpart.
    
    parameters
    ----------
    cpart : CanvasPart object
        The CanvasPart object for which we want to calculate the instability
    t_lims : 1d array-like of floats
        time intervals in which the pixel states are studied. Must be ordered in a crescent way, start at 0, and be regularly spaced
    testvariable : 1d array-like of floats
        The variable on which cutoffs will be applied to detect transitions
    cutoff : float
        the lower cutoff on instability to define when a transition is happening
    cutoff_stable : float
        the higher cutoff on instability, needs to be not exceeded before and after the potential transition, for a validated transition
    len_stableregion : float 
        the duration (in seconds), before and after the potential transition, during which cutoff_stable must not be exceeded. 
        No stable intervals is looked for if len_stableregion <= 1 or cutoff_stable == cutoff.
    distfromtrans_stableregion: float
        maximal distance (in seconds) between the borders of the stable regions and those of the transition region.
    sliding_win: float
        The time (in seconds) added to distfromtrans_stableregion for the post-transtion stable period,
        to take into account the post-transition adaptation of the reference image over the sliding window

    returns
    -------
    (full_transition, full_transition_times) :
        full_transition is a 2d array. 
        full_transition[i] contains, for the found transition #i, the following size-6 array (of indices of the input instability array) :
            [beginning of preceding stable region, end of stable, 
            beginning of transition region, end of transition, 
            beginning of subsequent stable region, end of stable]
        (NB: a region of indices [n, m] includes all time intervals from n to m included)
        full_transition_times is the same array but with the times corresponding to those indices 
    '''
    # preliminary
    len_stableregion = math.ceil(len_stableregion / (t_lims[1] - t_lims[0])) # assumes regularly-spaced t_lims
    distfromtrans_stableregion = math.ceil(distfromtrans_stableregion / (t_lims[1] - t_lims[0])) # assumes regularly-spaced t_lims
    if cutoff < cutoff_stable:
        raise ValueError("ERROR in find_transitions(): instability cutoff for transitions must be higher than that for stable periods")
    
    #print([(i,testvariable[i]) for i in range(0,len(testvariable))])
    # removing sequence of "1." or 0. at the beginning of testvariable
    ones_ind = np.where((testvariable == 1.) | (testvariable == 0.))[0]
    end_first_ones_sequence = -1
    if len(ones_ind) > 0:
        interruptions_ones_sequence = np.where( np.diff( ones_ind ) > 1)[0]
        end_first_ones_sequence = interruptions_ones_sequence[0] if len(interruptions_ones_sequence) > 0 else ones_ind[-1]
        testvariable = testvariable[(end_first_ones_sequence+1):] # remove elements until first sequence of ones is over

    # indices of testvariable that are stable or in transition
    trans_ind = np.where(np.array(testvariable) > cutoff)[0]

    stable_ind = np.where((np.array(testvariable) < cutoff_stable)
                       & ((np.array(testvariable) > 0) | (np.roll(testvariable, 1) > 0) | (np.roll(testvariable, 2) > 0)))[0]

    # get the sequences of at least len_stableregion consecutive indices in stable_ind
    if len_stableregion < 2:
        start_stable_inds = stable_ind
    elif (len(stable_ind) < len_stableregion - 1):
        start_stable_inds = np.array([])
    else:     
        stable_ind_to_sum = np.zeros( shape=(len_stableregion - 1, len(stable_ind) - len_stableregion + 1) , dtype=np.int16)
        for i in range(0, len_stableregion - 1):
            offset_ind = range(i, len(stable_ind) - len_stableregion + 2 + i) # sets of indices with crescent offsets
            stable_ind_to_sum[i] = np.diff( stable_ind[offset_ind] ) # difference between elements n and n-1 for these offsetted arrays

        start_stable_inds = stable_ind[ np.where( np.sum( stable_ind_to_sum, axis=0 ) == len_stableregion - 1 )[0] ]

    # only starting elements of uninterrupted sequences
    start_stable_inds_clean = 1 + np.where(np.diff(start_stable_inds) > 1)[0]
    if len(start_stable_inds) > 0:
        start_stable_inds_clean = np.concatenate(([0], start_stable_inds_clean)) # keep the first sequence
    start_sequence_stable_ind = start_stable_inds[start_stable_inds_clean]

    # end indices of stable periods
    end_stab_inds = np.hstack([(start_stable_inds_clean - 1)[1:], -1]) if len(start_stable_inds) > 0 else np.array([], dtype=np.int16)
    end_sequence_stable_ind = start_stable_inds[end_stab_inds] + len_stableregion - 1

    # get the start and end indices for transitions
    start_inds = 1 + np.where(np.diff(trans_ind) > 1)[0]
    end_inds = np.hstack([(start_inds - 1)[1:], -1]) # get rid of first value. Add an end value
    # keep only transitions that are surrounded by stable periods, 
    # and that the borders of the stable regions and those of the transition region are close enough
    start_inds_filtered = []
    end_inds_filtered = []
    stable_regions_borders = []
    for (s_ind, e_ind) in zip(start_inds, end_inds):
        trans_closeto_stable = np.where(  (end_sequence_stable_ind[:-1] < trans_ind[s_ind]) # exists a stable region before transition starts
                                        & (end_sequence_stable_ind[:-1] >= trans_ind[s_ind] - distfromtrans_stableregion) # at a close enough distance
                                        & (start_sequence_stable_ind[1:] > trans_ind[e_ind]) # stable region after the transition ends
                                        & (start_sequence_stable_ind[1:] <= trans_ind[e_ind] + distfromtrans_stableregion + sliding_win) # close enough
                                        )[0]
        if len(trans_closeto_stable) > 0:
            start_inds_filtered.append(s_ind)
            end_inds_filtered.append(e_ind)
            stable_regions_borders.append( [start_sequence_stable_ind[trans_closeto_stable][0], 
                                            end_sequence_stable_ind[trans_closeto_stable][0],
                                            start_sequence_stable_ind[np.array(trans_closeto_stable+1)][0], 
                                            end_sequence_stable_ind[np.array(trans_closeto_stable+1)][0]] )
            if len(trans_closeto_stable) > 1:
                print('\n !!!!!!!!!!!!!!!!!!!!!!!!!!', end_sequence_stable_ind[trans_closeto_stable], start_sequence_stable_ind[np.array(trans_closeto_stable+1)])
    
    # get the true start/end indices and times    
    trans_start_inds = trans_ind[np.array(start_inds_filtered, dtype=np.int16)]
    trans_end_inds = trans_ind[np.array(end_inds_filtered, dtype=np.int16)]

    # output
    full_transition_tmp = [ [s[0], s[1], t1, t2, s[2], s[3]] for (s, t1, t2) in zip(stable_regions_borders, trans_start_inds, trans_end_inds) ]
    full_transition = []
    # merge transitions that have the same preceding and subsequent stable regions
    deleted_idx = []
    for i in range(0, len(full_transition_tmp)):
        if i in deleted_idx: # this transition was previously deleted in the second-level loop
            break
        trans = full_transition_tmp[i]
        for j in range(i, len(full_transition_tmp)):
            trans2 = full_transition_tmp[j]
            if (trans[0] == trans2[0] and trans[1] == trans2[1] 
                and trans[4] == trans2[4] and trans[5] == trans2[5]):
                trans[2] = min(trans[2], trans2[2]) # new transition time interval is the smallest that contains both transition intervals
                trans[3] = max(trans[3], trans2[3])
                deleted_idx.append(j)
        full_transition.append(trans)

    # final output
    full_transition = np.array(full_transition + np.array(end_first_ones_sequence+1), dtype=np.int16) # add back the indices for the starting [1., 1., ...] sequence
    if len(full_transition) == 0:
        full_transition_times = np.array([])
    else: 
        full_transition_times = t_lims[full_transition + np.array([0, 1, 0, 1, 0, 1])]  # end time taken as the end of the last in-transition time interval

    return (full_transition, full_transition_times)

def transition_start_time(cpstat, tr):

    vars = [cpstat.n_changes_norm,
            cpstat.frac_attack_changes - 0.5,
            cpstat.n_users_norm,
            cpstat.returntime_median_overln2,
            cpstat.instability_norm,
            cpstat.frac_pixdiff_stable_vs_ref,
            cpstat.frac_pixdiff_inst_vs_stable_norm,
            1 - cpstat.frac_defenseonly_users - cpstat.frac_bothattdef_users - 0.5, #attackonly - 0.5
            cpstat.cumul_attack_timefrac,
            cpstat.entropy
            ]

    halfmax_times = np.zeros(len(vars))

    t_lims = cpstat.t_ranges
    # indices of the transition times
    trans_tind = cpstat.transition_tinds[tr]
    ind_st = trans_tind[1] - 1
    ind_end = trans_tind[3]

    for i in range(len(vars)):
        # max of vars over transition time range
        maxi = np.max(vars[i][ind_st : ind_end])
        # look for the time at which vars reaches halfmax value, starting search from end of pre-trans stable period
        halfmax_ind = ind_st + np.argmax(vars[i][ ind_st:ind_end ] > maxi/2)
        # use a linear interpolation of the values of the variable at each time step
        lin_interpol = lambda x: np.interp(x, t_lims[np.arange(halfmax_ind-1, halfmax_ind+1)], 
                                              vars[i][ (halfmax_ind-1):(halfmax_ind+1) ]  ) - maxi/2
        halfmax_times[i] = optimize.fsolve(lin_interpol, [t_lims[halfmax_ind-1]], xtol=5e-4)[0]

    mean_halfmaxtime = np.mean(halfmax_times)
    median_halfmaxtime = np.median(halfmax_times)
    rms_halfmaxtime = np.std(halfmax_times)
    #print(t_lims[halfmax_times], mean_halfmaxtime, rms_halfmaxtime, median_halfmaxtime)
    #print(cpstat.transition_times[tr][2:4])

    return halfmax_times, mean_halfmaxtime, median_halfmaxtime, rms_halfmaxtime
    