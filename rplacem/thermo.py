import json
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageColor
import Variables.Variables as var
import canvas_part as cp

def calc_num_pixel_changes(canvas_part,
                           time_inds_list,
                           time_interval):
    '''
    calculate several quantities related to the rate of pixel changes

    parameters
    ----------
    canvas_part : CanvasPart object
        The CanvasPart object for which we want to calculate the pixel change quantities
    time_inds_list : list of numpy arrays
        list containing arrays of the time indices integrated up to each time step
    time_interval : float
        time interval between frames in seconds

    returns
    -------
    num_pixel_changes : 1d numpy array (length of number of time steps)
        The number of pixel changes in the boundary since the start up to the time interval
    num_touched_pixels : 1d numpy array (length of number of time steps)
        The number of touched pixels (only one change is counted per pixel) since the start
        up to the time interval
    num_users: 1d numpy array (length: number of times steps)
        The number of active user at each time interval
    '''

    num_pixel_changes = np.zeros(len(time_inds_list))
    num_touched_pixels = np.zeros(len(time_inds_list))
    num_users = np.zeros(len(time_inds_list))
    coords_all = canvas_part.pixel_changes_coords()

    for i in range(0, len(time_inds_list)):

        # get number of pixel changes that have had at least one change
        # since the start

        # get the pixel change coordinates for the interval
        pixel_changes_time_int = canvas_part.pixel_changes[time_inds_list[i]]
        coords = coords_all[:, time_inds_list[i]]
        user_id = np.array(pixel_changes_time_int['user'])

        # get rid of the duplicate pixel changes
        unique_pixel_changes = np.unique(coords, axis=1)
        num_touched_pixels[i] = unique_pixel_changes.shape[1]

        # get the number of unique user ids
        unique_ids = np.unique(user_id)
        num_users[i] = unique_ids.shape[0]

        # number of pixel changes within the current time interval
        num_pixel_changes[i] = len(time_inds_list[i])

    return (num_pixel_changes,
            num_touched_pixels,
            num_users)

def find_transitions(t_lims,
                     stability_vs_time,
                     cutoff=0.88,
                     cutoff_stable=0.98,
                     num_stable_intervals=4,
                     dist_stableregion_transition=3
                     ):
    '''
    identifies transitions in a canvas_part.
    
    parameters
    ----------
    canvas_part : CanvasPart object
        The CanvasPart object for which we want to calculate the stability
    t_lims : 1d array-like of floats
        time intervals in which the pixel states are studied. Must be ordered in a crescent way, and start at 0
    stability_vs_time : 1d array-like of floats
        stability array averaged over all the pixels, for each time step
    cutoff : float
        the higher cutoff on stability to define when a transition is happening
    cutoff_stable : float
        the lower cutoff on stability, needs to be exceeded before and after the potential transition, for a validated transition
    num_stable_intervals : int
        the number of consecutive intervals before and after the potential transition, for which cutoff_stable must be exceeded. 
        No stable intervals is looked for if num_stable_intervals <= 1 or cutoff_stable == cutoff.
    dist_stableregion_transition: int
        maximal distance (in number of indices) between the borders of the stable regions and those of the transition region

    returns
    -------
    (full_transition, full_transition_times) :
        full_transition is a 2d array. 
        full_transition[i] contains, for the found transition #i, the following size-6 array (of indices of the input stability array) :
            [beginning of preceding stable region, end of stable, 
            beginning of transition region, end of transition, 
            beginning of subsequent stable region, end of stable]
        (NB: a region of indices [n, m] includes all time intervals from n to m included)
        full_transition_times is the same array but with the times corresponding to those indices 
    '''
    # preliminary
    if cutoff > cutoff_stable:
        raise ValueError("ERROR in find_transitions(): stability cutoff for transitions must be lower than that for stable periods")
    dist_stableregion_transition = max(dist_stableregion_transition, 1)
    
    #print([(i,stability_vs_time[i]) for i in range(0,len(stability_vs_time))])
    # removing sequence of "1." at the beginning of stability_vs_time
    ones_ind = np.where(stability_vs_time == 1.)[0]
    end_first_ones_sequence = -1
    if len(ones_ind) > 0:
        interruptions_ones_sequence = np.where( np.diff( ones_ind ) > 1)[0]
        end_first_ones_sequence = interruptions_ones_sequence[0] if len(interruptions_ones_sequence) > 0 else ones_ind[-1]
        stability_vs_time = stability_vs_time[(end_first_ones_sequence+1):] # remove elements until first sequence of ones is over

    # indices of stability_vs_time that are stable or in transition
    trans_ind = np.where(np.array(stability_vs_time) < cutoff)[0]
    stable_ind = np.where(np.array(stability_vs_time) > cutoff_stable)[0]

    # get the sequences of at least num_stable_intervals consecutive indices in stable_ind
    if num_stable_intervals < 2:
        start_stable_inds = stable_ind
    elif (len(stable_ind) < num_stable_intervals - 1):
        start_stable_inds = np.array([])
    else:     
        stable_ind_to_sum = np.zeros( shape=(num_stable_intervals - 1, len(stable_ind) - num_stable_intervals + 1) , dtype=np.int16)
        for i in range(0, num_stable_intervals - 1):
            offset_ind = range(i, len(stable_ind) - num_stable_intervals + 2 + i) # sets of indices with crescent offsets
            stable_ind_to_sum[i] = np.diff( stable_ind[offset_ind] ) # difference between elements n and n-1 for these offsetted arrays

        start_stable_inds = stable_ind[ np.where( np.sum( stable_ind_to_sum, axis=0 ) == num_stable_intervals - 1 )[0] ]

    # only starting elements of uninterrupted sequences
    start_stable_inds_clean = 1 + np.where(np.diff(start_stable_inds) > 1)[0]
    if len(start_stable_inds) > 0:
        start_stable_inds_clean = np.concatenate(([0], start_stable_inds_clean)) # keep the first sequence
    start_sequence_stable_ind = start_stable_inds[start_stable_inds_clean]

    # end indices of stable periods
    end_stab_inds = np.hstack([(start_stable_inds_clean - 1)[1:], -1]) if len(start_stable_inds) > 0 else np.array([], dtype=np.int16)
    end_sequence_stable_ind = start_stable_inds[end_stab_inds] + num_stable_intervals - 1

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

def stability(canvas_part,
              t_lims=[0, var.TIME_TOTAL],
              save_images=False,
              save_pickle=False,
              ):
    '''
    makes map of stability in some time range.
    Stability of a pixel is the fraction of time it spent in its 'favorite' color (meaning the color it was in for the most time)

    parameters
    ----------
    canvas_part : CanvasPart object
        The CanvasPart object for which we want to calculate the stability
    t_lims : 1d array-like of floats
        time intervals in which the pixel states are studied. Must be ordered in a crescent way, and start at 0

    returns
    -------
    stability_vs_time: 1d array-like of floats
        stability array averaged over all the pixels, for each time step
    The optionaly saved pickle file records the stable_timefraction (time fraction spent in the dominant color)
    and the stable_color (the dominant colors). They are indexed in the same way than canvas_part.coords.
    '''

    if t_lims[0] != 0:
        print('WARNING: this function \'stability\' is not meant to work from a lower time limit > 0 !!!')

    seconds = np.array(canvas_part.pixel_changes['seconds'])
    color = np.array(canvas_part.pixel_changes['color'])
    coord_inds = np.array(canvas_part.pixel_changes['coord_index']) # indices of canvas_part.coords where to find the (x,y) of a given pixel_change 
    num_coords = canvas_part.coords.shape[1]
    coord_range = np.arange(0, num_coords)

    quarter1_inds = np.where((canvas_part.coords[0]<1000) & (canvas_part.coords[1]<1000))
    quarter2_inds = np.where((canvas_part.coords[0]>=1000) & (canvas_part.coords[1]<1000))
    quarter34_inds = np.where(canvas_part.coords[1]>=1000)
    
    color_dict = json.load(open(os.path.join(canvas_part.data_path, 'ColorDict.json')))
    white = color_dict['#FFFFFF']
    current_color = np.full(num_coords, white, dtype='int8')
    stability_vs_time = np.zeros(len(t_lims)-1)

    for t_step in range(0, len(t_lims)-1):
        # get indices of all pixel changes that happen in the step
        t_inds = np.where((seconds >= t_lims[t_step]) & (seconds < t_lims[t_step+1]))[0]

        # time spent for each pixel in each of the 32 colors
        time_spent_in_color = np.zeros((num_coords, len(color_dict)), dtype='float64')

        last_time_changed = np.full(num_coords, t_lims[t_step], dtype='float64')
        # neglect the time before the opening of the supplementary canvas quarters
        last_time_changed[quarter2_inds] = max(t_lims[t_step], var.TIME_ENLARGE1)
        last_time_changed[quarter34_inds] = max(t_lims[t_step], var.TIME_ENLARGE2)

        # Can we find a way to get rid of this loop? -> not possible because the current_color must be changed before each iteration
        # TODO: add condition is_in_comp on keeping each pixel change
        for tidx in t_inds: # loop through each pixel change in the step, indexing by time
            s = seconds[tidx]
            c = color[tidx]
            coor_idx = coord_inds[tidx]

            # add the time that this pixel spent in the most recent color
            time_spent_in_color[coor_idx, current_color[coor_idx]] += s - last_time_changed[coor_idx]
            # time_spent_in_color[coord_inds[t_inds], current_color[coord_inds[t_inds]]] += seconds[t_inds] - last_time_changed[coord_inds[t_inds]]

            # update the time and color of the last pixel change for this pixel
            last_time_changed[coor_idx] = s  # last_time_changed[coord_inds[t_inds]] = seconds[t_inds] 

            # between each pixel change, there is only one color, until the time of the next change
            current_color[coor_idx] = c

        # add the time spent in the final color (from the last pixel change to the end-time)
        time_spent_in_color[coord_range, current_color] += np.maximum(t_lims[t_step+1] - last_time_changed, 0)

        # get the color where pixels spent the most time
        stable_colors = np.flip(np.argsort(time_spent_in_color, axis=1), axis=1)
        stable_timefraction = np.take_along_axis(time_spent_in_color, stable_colors, axis=1)
     
        # normalize by the total time the canvas quarter was on
        stable_timefraction[quarter1_inds] /= t_lims[t_step+1] - t_lims[t_step]
        stable_timefraction[quarter2_inds] /= t_lims[t_step+1] - max(t_lims[t_step], var.TIME_ENLARGE1)
        stable_timefraction[quarter34_inds] /= t_lims[t_step+1] - max(t_lims[t_step], var.TIME_ENLARGE2)

        ##############
        # calculate the stability averaged over the whole canvas_part
        stab_per_pixel = stable_timefraction[:, 0] # get time fraction for most stable pixel
        inds_nonzero = np.where(stab_per_pixel>1e-10) # will remove indices with stab==0

        # get indices of coordinates for which the interval [t_lims[t_step], t_lims[t_step+1]] intersects with the 'active' timerange for the composition
        inds_in_timerange = np.arange(0, num_coords)
        if not canvas_part.is_rectangle:
            timeranges = canvas_part.coords_timerange
            if timeranges.dtype != object: # case where the timeranges array is a numpy array, with only 1 timerange per pixel
                # tried to reduce this through distributivity of logical operations, but inconclusive
                inds_in_timerange = np.where( ((t_lims[t_step] >= timeranges[:,0,0]) | (t_lims[t_step+1] > timeranges[:,0,0])) 
                                            & ((t_lims[t_step] < timeranges[:,0,1]) | (t_lims[t_step+1] <= timeranges[:,0,1])) )
            else: # dirty python here: timeranges is not a full numpy array, so cannot use broadcasting
                inds_in_timerange = []
                for i in range(0, num_coords):
                    in_timerange = False
                    for timerange in timeranges[i]:
                        in_timerange |= (((t_lims[t_step] >= timerange[0]) | (t_lims[t_step+1] > timerange[0])) 
                                       & ((t_lims[t_step] < timerange[1]) | (t_lims[t_step+1] <= timerange[1])) )
                    if in_timerange:
                        inds_in_timerange.append(i)
                inds_in_timerange = np.array(inds_in_timerange, dtype=int)

        stab_per_pixel = stab_per_pixel[np.intersect1d(inds_nonzero, inds_in_timerange, assume_unique=True)] 
        # now actually get stability averaged over pixels
        if len(stab_per_pixel) > 0:
            stability = np.mean(stab_per_pixel)
        else:
            stability = 1
        stability_vs_time[t_step] = stability
        #############

        # save full result to pickle file
        if save_pickle:
            file_path = os.path.join(os.path.join(os.getcwd(), 'data'), 'stability_' + canvas_part.out_name() + '_time{:06d}to{:06d}.pickle'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
            with open(file_path, 'wb') as handle:
                pickle.dump([stable_timefraction,
                            stable_colors],
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        # create images containing the (sub)dominant color only
        if save_images:
            pixels1 = np.full((canvas_part.ymax - canvas_part.ymin + 1, canvas_part.xmax - canvas_part.xmin + 1, 3), 255, dtype=np.uint8)
            pixels2 = np.full((canvas_part.ymax - canvas_part.ymin + 1, canvas_part.xmax - canvas_part.xmin + 1, 3), 255, dtype=np.uint8)
            pixels3 = np.full((canvas_part.ymax - canvas_part.ymin + 1, canvas_part.xmax - canvas_part.xmin + 1, 3), 255, dtype=np.uint8)

            pixels1[canvas_part.coords[1]-canvas_part.ymin, canvas_part.coords[0]-canvas_part.xmin] = canvas_part.get_rgb(stable_colors[:,0])
            pixels2[canvas_part.coords[1]-canvas_part.ymin, canvas_part.coords[0]-canvas_part.xmin] = canvas_part.get_rgb(stable_colors[:,1])
            pixels3[canvas_part.coords[1]-canvas_part.ymin, canvas_part.coords[0]-canvas_part.xmin] = canvas_part.get_rgb(stable_colors[:,2])

            # if second and/or third most used colors don't exist (time_spent == 0), then use the first or second most used color instead
            inds_to_change1 = np.where(stable_timefraction[:,1] < 1e-9)
            pixels2[canvas_part.coords[1,inds_to_change1] - canvas_part.ymin, 
                    canvas_part.coords[0,inds_to_change1]-canvas_part.xmin] = pixels1[canvas_part.coords[1,inds_to_change1]-canvas_part.ymin, 
                                                                                      canvas_part.coords[0,inds_to_change1]-canvas_part.xmin]
            inds_to_change2 = np.where(stable_timefraction[:,2] < 1e-9)
            pixels3[canvas_part.coords[1,inds_to_change2] - canvas_part.ymin, 
                    canvas_part.coords[0,inds_to_change2]-canvas_part.xmin] = pixels2[canvas_part.coords[1,inds_to_change2]-canvas_part.ymin, 
                                                                                      canvas_part.coords[0,inds_to_change2]-canvas_part.xmin]

            # save images
            try:
                os.makedirs(os.path.join(os.getcwd(), 'figs', 'history_' + canvas_part.out_name()))
            except OSError: 
                pass
            im1 = Image.fromarray(pixels1.astype(np.uint8))
            im1_path = os.path.join(os.getcwd(), 'figs','history_' + canvas_part.out_name(), 'MostStableColor_time{:06d}to{:06d}.png'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
            im1.save(im1_path)
            im2 = Image.fromarray(pixels2.astype(np.uint8))
            im2_path = os.path.join(os.getcwd(), 'figs','history_' + canvas_part.out_name(), 'SecondMostStableColor_time{:06d}to{:06d}.png'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
            im2.save(im2_path)
            im3 = Image.fromarray(pixels3.astype(np.uint8))
            im3_path = os.path.join(os.getcwd(), 'figs','history_' + canvas_part.out_name(), 'ThirdMostStableColor_time{:06d}to{:06d}.png'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
            im3.save(im3_path)
    
    return stability_vs_time

def plot_compression_vs_pixel_changes(num_pixel_changes,
                                      num_touched_pixels,
                                      num_users,
                                      times,
                                      file_size_png,
                                      file_size_bmp):
    '''
    plot the pixel change quanities and CID (computable information density)
    '''

    fig_cid_vs_time = plt.figure()
    plt.plot(times, file_size_png/file_size_bmp)
    plt.ylabel('Computational Information Density  (file size ratio)')
    plt.xlabel('Time (s)')
    sns.despine()

    fig_num_touched_pix_vs_time = plt.figure()
    plt.plot(times, num_touched_pixels)
    plt.ylabel('Number of touched pixels')
    plt.xlabel('Time (s)')
    sns.despine()

    fig_num_pix_changes_vs_time = plt.figure()
    plt.plot(times, num_pixel_changes)
    plt.ylabel('Number of Pixel Changes')
    plt.xlabel('Time (s)')
    sns.despine()

    fig_users_vs_time = plt.figure()
    plt.plot(times, num_users)
    plt.ylabel('Number of Users')
    plt.xlabel('Time (s)')
    sns.despine()

    fig_cid_vs_num_pix_changes = plt.figure()
    plt.scatter(num_pixel_changes, file_size_png/file_size_bmp, s=5, alpha=0.7, c=times)
    plt.xlabel('Number of Pixel Changes')
    plt.ylabel('Computational Information Density (file size ratio)')
    sns.despine()

    fig_cid_vs_num_touched_pix = plt.figure()
    plt.scatter(num_touched_pixels, file_size_png/file_size_bmp, s=5, alpha=0.7, c=times)
    plt.xlabel('Number of touched Pixels')
    plt.ylabel('Computational Information Density (file size ratio)')
    sns.despine()

    fig_cid_vs_num_users = plt.figure()
    plt.scatter(num_users, file_size_png/file_size_bmp, s=5, alpha=0.7, c=times)
    plt.xlabel('Number of Users')
    plt.ylabel('Computational Information Density (file size ratio)')
    sns.despine()

    return (fig_cid_vs_time,
            fig_num_touched_pix_vs_time,
            fig_num_pix_changes_vs_time,
            fig_users_vs_time,
            fig_cid_vs_num_pix_changes,
            fig_cid_vs_num_touched_pix,
            fig_cid_vs_num_users)

