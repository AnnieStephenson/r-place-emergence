import numpy as np
import os
import rplacem.canvas_part as cp
import rplacem.compute_variables as comp
import rplacem.utilities as util
import rplacem.variables_rplace2022 as var
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def find_transitions(t_lims,
                     instability_vs_time,
                     cutoff=7e-3, 
                     cutoff_stable=1.5e-3,
                     len_stable_intervals=4,
                     dist_stableregion_transition=3
                     ):
    '''
    identifies transitions in a cpart.
    
    parameters
    ----------
    cpart : CanvasPart object
        The CanvasPart object for which we want to calculate the instability
    t_lims : 1d array-like of floats
        time intervals in which the pixel states are studied. Must be ordered in a crescent way, and start at 0
    instability_vs_time : 1d array-like of floats
        instability array averaged over all the pixels and normalized, for each time step
    cutoff : float
        the lower cutoff on instability to define when a transition is happening
    cutoff_stable : float
        the higher cutoff on instability, needs to be not exceeded before and after the potential transition, for a validated transition
    len_stable_intervals : int
        the number of consecutive intervals before and after the potential transition, for which cutoff_stable must not be exceeded. 
        No stable intervals is looked for if len_stable_intervals <= 1 or cutoff_stable == cutoff.
    dist_stableregion_transition: int
        maximal distance (in number of indices) between the borders of the stable regions and those of the transition region

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
    if cutoff < cutoff_stable:
        raise ValueError("ERROR in find_transitions(): instability cutoff for transitions must be higher than that for stable periods")
    dist_stableregion_transition = max(dist_stableregion_transition, 1)
    
    #print([(i,instability_vs_time[i]) for i in range(0,len(instability_vs_time))])
    # removing sequence of "1." at the beginning of instability_vs_time
    ones_ind = np.where(instability_vs_time == 1.)[0]
    end_first_ones_sequence = -1
    if len(ones_ind) > 0:
        interruptions_ones_sequence = np.where( np.diff( ones_ind ) > 1)[0]
        end_first_ones_sequence = interruptions_ones_sequence[0] if len(interruptions_ones_sequence) > 0 else ones_ind[-1]
        instability_vs_time = instability_vs_time[(end_first_ones_sequence+1):] # remove elements until first sequence of ones is over

    # indices of instability_vs_time that are stable or in transition
    trans_ind = np.where(np.array(instability_vs_time) > cutoff)[0]

    stable_ind = np.where((np.array(instability_vs_time) < cutoff_stable)
                       & ((np.array(instability_vs_time) > 0) | (np.roll(instability_vs_time, 1) > 0) | (np.roll(instability_vs_time, 2) > 0)))[0]

    # get the sequences of at least len_stable_intervals consecutive indices in stable_ind
    if len_stable_intervals < 2:
        start_stable_inds = stable_ind
    elif (len(stable_ind) < len_stable_intervals - 1):
        start_stable_inds = np.array([])
    else:     
        stable_ind_to_sum = np.zeros( shape=(len_stable_intervals - 1, len(stable_ind) - len_stable_intervals + 1) , dtype=np.int16)
        for i in range(0, len_stable_intervals - 1):
            offset_ind = range(i, len(stable_ind) - len_stable_intervals + 2 + i) # sets of indices with crescent offsets
            stable_ind_to_sum[i] = np.diff( stable_ind[offset_ind] ) # difference between elements n and n-1 for these offsetted arrays

        start_stable_inds = stable_ind[ np.where( np.sum( stable_ind_to_sum, axis=0 ) == len_stable_intervals - 1 )[0] ]

    # only starting elements of uninterrupted sequences
    start_stable_inds_clean = 1 + np.where(np.diff(start_stable_inds) > 1)[0]
    if len(start_stable_inds) > 0:
        start_stable_inds_clean = np.concatenate(([0], start_stable_inds_clean)) # keep the first sequence
    start_sequence_stable_ind = start_stable_inds[start_stable_inds_clean]

    # end indices of stable periods
    end_stab_inds = np.hstack([(start_stable_inds_clean - 1)[1:], -1]) if len(start_stable_inds) > 0 else np.array([], dtype=np.int16)
    end_sequence_stable_ind = start_stable_inds[end_stab_inds] + len_stable_intervals - 1

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
                                        & (end_sequence_stable_ind[:-1] >= trans_ind[s_ind] - dist_stableregion_transition) # at a close enough distance
                                        & (start_sequence_stable_ind[1:] > trans_ind[e_ind]) # stable region after the transition ends
                                        & (start_sequence_stable_ind[1:] >= trans_ind[e_ind] - dist_stableregion_transition) # close enough
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

def transition_and_reference_image(canpart, 
                               time_ranges, 
                               instability_vs_time,
                               create_refimages,
                               save_images,
                               averaging_period=3600, # 1 hour
                               cutoff=7e-3,
                               cutoff_stable=1.5e-3,
                               len_stable_intervals=3,
                               dist_stableregion_transition=3
                               ):
    ''' Finds transitions for a given canvas part, 
    and returns the images containing the most stable pixels for 
    the stable periods before and after the transition, and during the transition.
    [instability_vs_time] input must be the one corresponding to the input canvas_part.
    [averaging_period] is the time over which the reference images before and after the transition are computed.
    '''
    
    transitions = find_transitions(time_ranges, instability_vs_time, 
                                   cutoff, cutoff_stable, len_stable_intervals, dist_stableregion_transition)
    avimage_pre = []
    avimage_trans = []
    avimage_post = []
    num_differing_pixels = []

    if create_refimages:
        for j in range(0, len(transitions[0])):
            trans_times2 = np.hstack((0, transitions[1][j]))
            trans_times2[1] = trans_times2[2] - averaging_period # calculate the (pre)stable image in only the latest stable time interval 
            trans_times2[6] = trans_times2[5] + averaging_period # calculate the (post)stable image in only the earliest stable time interval 
            _, _, stablepixels1, _, _, _ = comp.stability(canpart, trans_times2, True,  False, False, False, False)
            avimage_pre.append(stablepixels1[1])
            avimage_trans.append(stablepixels1[3])
            avimage_post.append(stablepixels1[5])
            if save_images:
                util.pixels_to_image(avimage_pre[j], canpart.out_name(), 'MostStableColor_referenceimage_pre_transition'+str(j) + '.png')
                util.pixels_to_image(avimage_trans[j], canpart.out_name(), 'MostStableColor_referenceimage_during_transition'+str(j) + '.png')
                util.pixels_to_image(avimage_post[j], canpart.out_name(), 'MostStableColor_referenceimage_post_transition'+str(j) + '.png')

            num_differing_pixels.append( comp.count_image_differences(avimage_post[j], avimage_pre[j], canpart) )

    return (avimage_pre, avimage_trans, avimage_post, 
            num_differing_pixels, transitions[0], transitions[1])