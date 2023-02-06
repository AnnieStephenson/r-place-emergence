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
                     len_stable_intervals=4,
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
    len_stable_intervals : int
        the number of consecutive intervals before and after the potential transition, for which cutoff_stable must be exceeded. 
        No stable intervals is looked for if len_stable_intervals <= 1 or cutoff_stable == cutoff.
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

def count_image_differences(pixels1, pixels2, cpart, indices=None):
    ''' Count the number of pixels (at given *indices* of coordinates of canvas_part *cpart*) that differ 
    between *pixels1* and *pixels2* (both 2d numpy arrays of shape (num y coords, num x coords))'''
    if indices is None:
        indices = np.arange(0, cpart.coords.shape[1])
    ycoor = cpart.coords[1, indices] - cpart.ymin
    xcoor = cpart.coords[0, indices] - cpart.xmin
    return np.count_nonzero(pixels2[ycoor, xcoor] - pixels1[ycoor, xcoor])

def pixels_to_image(pix, canvas_part=None, save_name=''):
    ''' Transform the 2d array of color indices into an image object, and saves it if (save_name!='') '''
    im = Image.fromarray( canvas_part.get_rgb(pix).astype(np.uint8) )
    if save_name != '':
        if canvas_part == None:
            im.save( save_name )
        else:
            try:
                os.makedirs( os.path.join(var.FIGS_PATH, 'history_' + canvas_part.out_name()) )
            except OSError: 
                pass
            im.save( os.path.join(var.FIGS_PATH, 'history_' + canvas_part.out_name(), save_name) )

def stability(canvas_part,
              t_lims=[0, var.TIME_TOTAL],
              create_images=False,
              save_images=False,
              save_pickle=False,
              compute_average=True
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
    save_images, save_pickle, compute_average: bools
        flags, see 'returns' section

    returns
    -------
    stability_vs_time: 1d array-like of floats
        stability array averaged over all the pixels, for each time step
        Calculated only if compute_average==True
    The saved pickle file (if save_pickle==True) records the stable_timefraction (time fraction spent in the dominant color)
        and the stable_color (the dominant colors). They are indexed in the same way than canvas_part.coords.
    The images (computed if create_images==True, saved if save_images==True) show 
        the most stable color of each pixel during each time interval.
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
    
    current_color = np.full(num_coords, var.WHITE, dtype='int8')
    stability_vs_time = np.zeros(len(t_lims)-1)
    if create_images:
        # 2d numpy arrays containing color indices (from 0 to 31) for each pixel of the composition
        pixels1 = np.full((len(t_lims)-1, canvas_part.ymax - canvas_part.ymin + 1, canvas_part.xmax - canvas_part.xmin + 1), var.WHITE, dtype=np.int8)
        pixels2 = np.full((len(t_lims)-1, canvas_part.ymax - canvas_part.ymin + 1, canvas_part.xmax - canvas_part.xmin + 1), var.WHITE, dtype=np.int8)
        pixels3 = np.full((len(t_lims)-1, canvas_part.ymax - canvas_part.ymin + 1, canvas_part.xmax - canvas_part.xmin + 1), var.WHITE, dtype=np.int8)


    for t_step in range(0, len(t_lims)-1):
        # get indices of all pixel changes that happen in the step
        t_inds = np.where((seconds >= t_lims[t_step]) & (seconds < t_lims[t_step+1]))[0]

        # time spent for each pixel in each of the 32 colors
        time_spent_in_color = np.zeros((num_coords, var.NUM_COLORS), dtype='float64')

        last_time_changed = np.full(num_coords, t_lims[t_step], dtype='float64')
        # neglect the time before the opening of the supplementary canvas quarters
        last_time_changed[quarter2_inds] = max(t_lims[t_step], var.TIME_ENLARGE1)
        last_time_changed[quarter34_inds] = max(t_lims[t_step], var.TIME_ENLARGE2)

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
        if compute_average:
            stab_per_pixel = stable_timefraction[:, 0] # get time fraction for most stable pixel
            inds_nonzero = np.where(stab_per_pixel>1e-10) # will remove indices with stab==0

            # get indices of coordinates for which the interval [t_lims[t_step], t_lims[t_step+1]] intersects with the 'active' timerange for the composition
            inds_in_timerange = canvas_part.coord_indices_in_timerange(t_lims[t_step], t_lims[t_step+1])

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
        if create_images:
            pixels1[t_step, canvas_part.coords[1]-canvas_part.ymin, canvas_part.coords[0]-canvas_part.xmin] = stable_colors[:,0]
            pixels2[t_step, canvas_part.coords[1]-canvas_part.ymin, canvas_part.coords[0]-canvas_part.xmin] = stable_colors[:,1]
            pixels3[t_step, canvas_part.coords[1]-canvas_part.ymin, canvas_part.coords[0]-canvas_part.xmin] = stable_colors[:,2]

            # if second and/or third most used colors don't exist (time_spent == 0), then use the first or second most used color instead
            inds_to_change1 = np.where(stable_timefraction[:,1] < 1e-9)
            pixels2[t_step, canvas_part.coords[1,inds_to_change1] - canvas_part.ymin, 
                    canvas_part.coords[0,inds_to_change1]-canvas_part.xmin] = pixels1[t_step, canvas_part.coords[1,inds_to_change1]-canvas_part.ymin, 
                                                                                      canvas_part.coords[0,inds_to_change1]-canvas_part.xmin]
            inds_to_change2 = np.where(stable_timefraction[:,2] < 1e-9)
            pixels3[t_step, canvas_part.coords[1,inds_to_change2] - canvas_part.ymin, 
                    canvas_part.coords[0,inds_to_change2]-canvas_part.xmin] = pixels2[t_step, canvas_part.coords[1,inds_to_change2]-canvas_part.ymin, 
                                                                                      canvas_part.coords[0,inds_to_change2]-canvas_part.xmin]

            # save images
            if save_images:
                timerange_str = 'time{:06d}to{:06d}.png'.format(int(t_lims[t_step]), int(t_lims[t_step+1]))
                pixels_to_image( pixels1[t_step], canvas_part,
                                 'MostStableColor_' + timerange_str )
                pixels_to_image( pixels1[t_step], canvas_part,
                                 'SecondMostStableColor_' + timerange_str )
                pixels_to_image( pixels1[t_step], canvas_part,
                                 'ThirdMostStableColor_' + timerange_str )
    
    if not compute_average:
        stability_vs_time = np.array([])
    if not create_images:
        (pixels1, pixels2, pixels3) = (np.array([]), np.array([]), np.array([]))

    return (stability_vs_time, pixels1, pixels2, pixels3)


def num_deviating_pixels(canvas_part, t_lims, ref_image):
    '''
    Gives the number of pixel changes that agree or disagree with the provided reference (stable) image

    parameters
    ----------
    canvas_part : CanvasPart object
        The CanvasPart object under study
    t_lims : 1d array-like of floats
        time intervals in which the pixel states are studied. Must be ordered in a crescent way, and start at 0
    ref_image: 2d array, shape (# y coords, # of x coords)
        Image to which pixels at every time step are compared

    returns
    -------
    num_changes, num_attack_changes, num_defense_changes:
        the number of total pixel changes, those that restore the ref_image (defense), and those that destroy the ref_image (attack).
    num_active_coords:
        Active area of the composition in each timerange
    '''

    if t_lims[0] != 0:
        print('WARNING: this function \'stability\' is not meant to work from a lower time limit > 0 !!!')

    seconds = np.array(canvas_part.pixel_changes['seconds'])
    color = np.array(canvas_part.pixel_changes['color'])
    in_timerange = np.array(canvas_part.pixel_changes['in_timerange'])
    pix_change_coords = canvas_part.pixel_changes_coords()

    num_changes = np.zeros(len(t_lims)-1)
    num_defense_changes = np.zeros(len(t_lims)-1)
    num_active_coords = np.zeros(len(t_lims)-1)
    num_differing_pixels = np.zeros(len(t_lims)-1)

    current_image = np.full((canvas_part.ymax - canvas_part.ymin + 1, canvas_part.xmax - canvas_part.xmin + 1), var.WHITE, dtype=np.int8)
    for t_step in range(0, len(t_lims)-1):
        # get indices of all pixel changes that happen in the step (and that are part of the composition timerange for this pixel)
        t_inds = np.where((seconds >= t_lims[t_step]) & (seconds < t_lims[t_step+1]) & in_timerange)[0]
        t_inds_active = t_inds[ np.where(in_timerange[t_inds])[0] ]
        num_changes[t_step] = len(t_inds_active)

        # test if the color of this pixel change agrees with the reference image (ie is it of the same color)
        ycoords = np.array(pix_change_coords[1], np.int16) - canvas_part.ymin
        xcoords = np.array(pix_change_coords[0], np.int16) - canvas_part.xmin
        agreeing_changes = np.array( ref_image[ycoords[t_inds_active], xcoords[t_inds_active]] == color[t_inds_active] , np.bool_)
        num_defense_changes[t_step] = np.count_nonzero(agreeing_changes)

        # Update current_image with the pixel changes in this time interval. For each (x,y) pixel, need to keep only the last pixel change.
        color_inv = (color[t_inds])[::-1]
        ycoords_inv = (ycoords[t_inds])[::-1]
        xcoords_inv = (xcoords[t_inds])[::-1]
        xycoords_inv = np.column_stack((xcoords_inv, ycoords_inv))
        xyidx_unique, idx_first = np.unique(xycoords_inv, return_index=True, axis=0) # keeping the index of the first occurence in the reverse-order array
        current_image[xyidx_unique[:,1] , xyidx_unique[:,0]] = color_inv[idx_first]

        # count active (ie in timerange) coordinates and the differences with the ref_image at these coordinates
        active_coords_inds = canvas_part.coord_indices_in_timerange(t_lims[t_step], t_lims[t_step+1])
        num_active_coords[t_step] = len(active_coords_inds)
        num_differing_pixels[t_step] = count_image_differences(current_image, ref_image, canvas_part, active_coords_inds)

    num_attack_changes = num_changes - num_defense_changes
    return (num_changes, num_defense_changes, num_attack_changes, num_active_coords, num_differing_pixels)

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

