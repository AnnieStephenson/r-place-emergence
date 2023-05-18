import numpy as np
import rplacem.canvas_part as cp
import rplacem.compute_variables as comp
import rplacem.utilities as util
import rplacem.variables_rplace2022 as var
import math
from scipy import optimize

def limits_sequence_of_true(cond, min_len_seq):
    '''
    Gives the limits (indices) of sequences of True values from a boolean array

    parameters
    ----------
    cond: boolean array
        array of conditions that needs to be true between the beginning and end indices that are output
    min_len_seq: int
        minimum length of the sequences that will be kept

    returns
    -------
    beg
    end: int array of length n_sequences_output
        contains the beginning/end indices of wanted sequences
    '''
    cond_fenced = np.concatenate(([0], cond, [0]))
    lim_seq = np.flatnonzero(np.diff(cond_fenced) != 0) # limits of sequences of "True"
    beg_seq_true = lim_seq[::2] # beginning of sequences
    end_seq_true = lim_seq[1::2] # ends of sequences

    # keep only long enough sequences
    if min_len_seq > 1:
        long_enough = (end_seq_true - beg_seq_true >= min_len_seq)
        beg_seq_true = beg_seq_true[long_enough]
        end_seq_true = end_seq_true[long_enough]

    return np.column_stack((beg_seq_true, end_seq_true))

def merge_similar_transitions(transitions):
    '''
    Merge the transitions if they have the same pre- and post-transition periods 
    '''
    deleted_idx = []
    trans_merged = []
     
    for i in range(0, len(transitions)):
        if i in deleted_idx: # this transition was previously deleted in the second-level loop
            continue
        trans = transitions[i]
        for j in range(i, len(transitions)):
            trans2 = transitions[j]
            if (trans[0] == trans2[0] and trans[1] == trans2[1] 
                and trans[4] == trans2[4] and trans[5] == trans2[5]):
                trans[2] = min(trans[2], trans2[2]) # new transition time interval is the smallest that contains both transition intervals
                trans[3] = max(trans[3], trans2[3])
                deleted_idx.append(j)
        trans_merged.append(trans)

    return trans_merged

def find_transitions(t_lims,
                     testvar_pre, testvar_post,
                     cutoff=0.2, 
                     cutoff_stable=0.05,
                     len_stableregion=2*3600,
                     distfromtrans_stableregion=3*3600
                     ):
    '''
    identifies transitions in a cpart.
    
    parameters
    ----------
    cpart : CanvasPart object
        The CanvasPart object for which we want to calculate the instability
    t_lims : 1d array-like of floats
        time intervals in which the pixel states are studied. Must be ordered in a crescent way, start at 0, and be regularly spaced
    testvar_pre : 1d array-like of floats
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

    # transition and pre and post-transition stability conditions
    trans_cond = (np.array(testvar_pre) > cutoff)
    testvar_pre_cond = (np.array(testvar_pre) < cutoff_stable)
    testvar_post_cond = (np.array(testvar_post) < cutoff_stable) 

    # get the beg and end of sequences of at least len_stableregion indices that pass conditions
    seq_trans = limits_sequence_of_true(trans_cond, 1)
    seq_pretrans = limits_sequence_of_true(testvar_pre_cond, len_stableregion)
    seq_posttrans = limits_sequence_of_true(testvar_post_cond, len_stableregion)

    # keep only transitions that are surrounded by close-enough stable periods
    full_transition_tmp = []
    for tr in seq_trans:
        # stable sequences close enough and before transition
        pretrans_stable_seq = np.where((seq_pretrans[:, 1] <= tr[0])
                                     & (seq_pretrans[:, 1] > tr[0] - distfromtrans_stableregion))[0]
        # stable sequences close enough and after transition
        posttrans_stable_seq = np.where((seq_posttrans[:, 0] >= tr[1])
                                      & (seq_posttrans[:, 0] < tr[1] + distfromtrans_stableregion))[0]
        
        # keep transitions only if pre- and post-transition sequences are found
        if len(pretrans_stable_seq) > 0 and len(posttrans_stable_seq) > 0:
            full_transition_tmp.append(np.hstack( (seq_pretrans[pretrans_stable_seq[-1]], tr, seq_posttrans[posttrans_stable_seq[0]]) ))
    
    # merge transitions that have the same preceding and subsequent stable regions
    full_transitions = merge_similar_transitions(np.array(full_transition_tmp))
    return (full_transitions, t_lims[full_transitions])

def transition_start_time(cpstat, tr):

    vars = [cpstat.n_changes_norm.val,
            cpstat.frac_attack_changes.val - 0.5,
            cpstat.n_users_norm.val,
            cpstat.returntime_median_overln2.val,
            cpstat.instability_norm.val,
            cpstat.frac_pixdiff_stable_vs_swref.val,
            cpstat.frac_pixdiff_inst_vs_stable_norm.val,
            1 - cpstat.frac_defenseonly_users.val - cpstat.frac_bothattdef_users.val - 0.5, #attackonly - 0.5
            cpstat.cumul_attack_timefrac.val,
            cpstat.entropy.val
            ]

    halfmax_times = np.zeros(len(vars))

    t_lims = cpstat.t_lims
    # indices of the transition times
    trans_tind = cpstat.transition_tinds[tr]
    ind_st = trans_tind[1] - 1
    ind_end = trans_tind[3] + 1

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
    