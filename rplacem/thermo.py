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
from . import canvas_part as cp

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
    for i in range(0, len(time_inds_list)):

        # get number of pixel changes that have had at least one change
        # since the start

        # get the pixel change coordinates for the interval
        pixel_changes_time_int = canvas_part.pixel_changes[time_inds_list[i]]
        x_coord = np.array(pixel_changes_time_int['xcoor'])
        y_coord = np.array(pixel_changes_time_int['ycoor'])
        user_id = np.array(pixel_changes_time_int['user'])
        coords = np.vstack((x_coord, y_coord))

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

def find_transitions(canvas_part,
                     t_lims,
                     stability_vs_time,
                     cutoff=0.95):
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
        the cutoff stability to define when a transition is happening

    returns
    -------
    (trans_ind, trans_start_inds, trans_end_inds,
    trans_times, trans_start_times, trans_end_times, 
    num_trans, trans_durations):
       tuple of many properties of the transition time ranges, their indices, the numner of transitions, and 
       how long they last
    '''
    
    trans_ind = np.where(stability_vs_time < 0.95)[0]
    trans_times = t_lims[trans_ind + 1]

    # get the start indices and times
    start_inds = 1 + np.where(np.diff(trans_ind)>1)[0]
    trans_start_inds = trans_ind[start_inds]
    trans_start_times = trans_times[start_inds]
    
    # get the end indices and times
    end_inds = np.hstack([np.where(np.diff(trans_ind) > 1)[0][1:], -1]) # get rid of first value. Add on end value
    trans_end_inds = trans_ind[end_inds]
    trans_end_times = trans_times[end_inds]
    
    # get total number of transitions and their durations
    num_trans = len(trans_start_times)
    trans_durations = trans_end_times-trans_start_times

    return (trans_ind[1:], trans_start_inds, trans_end_inds,
           trans_times[1:], trans_start_times, trans_end_times, 
           num_trans, trans_durations)

def stability_new(canvas_part,
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
    '''

    if t_lims[0] != 0:
        print('WARNING: this function \'stability\' is not meant to work from a lower time limit > 0 !!!')

    seconds = np.array(canvas_part.pixel_changes['seconds'])
    color = np.array(canvas_part.pixel_changes['color'])
    num_coords = canvas_part.coords.shape[1]
    coord_range = np.arange(0,num_coords)

    quarter1_inds = np.where((canvas_part.coords[0]<1000) & (canvas_part.coords[1]<1000))
    quarter2_inds = np.where((canvas_part.coords[0]>1000) & (canvas_part.coords[1]<1000))
    quarter34_inds = np.where(canvas_part.coords[1]>1000)
    
    # get indices of canvas_part.coords where to find the x,y of a given pixel_change 
    coords_comb = canvas_part.coords[0] + 10000.*canvas_part.coords[1]
    coord_sort_inds = np.argsort(coords_comb)
    inds_in_sorted_coords = np.searchsorted(coords_comb[coord_sort_inds], canvas_part.pixel_changes['xcoor'] + 10000.*canvas_part.pixel_changes['ycoor'])
    coord_inds = coord_sort_inds[inds_in_sorted_coords]

    color_dict = json.load(open(os.path.join(canvas_part.data_path, 'ColorDict.json')))
    white = color_dict['#FFFFFF']
    current_color = np.full(num_coords,white, dtype='int8')
    stability_vs_time = np.zeros(len(t_lims)-1)

    for t_step in range(0, len(t_lims)-1):
        # get time indices of all pixel changes that happen in the step
        t_inds = np.where((seconds >= t_lims[t_step]) & (seconds < t_lims[t_step+1]))[0] #  & (xcoor >= xmin) & (xcoor <= xmax) & (ycoor >= ymin) & (ycoor <= ymax)

        # time spent for each pixel in each of the 32 colors
        time_spent_in_color = np.zeros((num_coords, len(color_dict)), dtype='float64')

        last_time_changed = np.full(num_coords, t_lims[t_step], dtype='float64')
       
        # neglect the time before the opening of the supplementary canvas quarters
        last_time_changed[quarter2_inds] = max(t_lims[t_step], var.TIME_ENLARGE1)
        last_time_changed[quarter34_inds] = max(t_lims[t_step], var.TIME_ENLARGE2)

        # TODO: can we find a way to get rid of this loop?
        for tidx in t_inds: # loop through each pixel change in the step, indexing by time
            s = seconds[tidx]
            c = color[tidx]

            # add the time that this pixel spent in the most recent color
            time_spent_in_color[coord_inds[tidx], current_color[coord_inds[tidx]]] += s - last_time_changed[coord_inds[tidx]]

            # update the time and color of the last pixel change for this pixel
            last_time_changed[coord_inds[tidx]] = s

            # between each pixel change, there is only one color, until the time of the next change
            current_color[coord_inds[tidx]] = c

        # add the time spent in the final color (from the last pixel change to the end-time)
        time_spent_in_color[coord_range, current_color] += np.maximum(t_lims[t_step+1] - last_time_changed, 0)

        # get the color where pixels spent the most time
        stable_colors = np.flip(np.argsort(time_spent_in_color, axis=1), axis=1)
        stable_timefraction = np.take_along_axis(time_spent_in_color, stable_colors, axis=1)
     
        # normalize by the total time the canvas quarter was on
        stable_timefraction[quarter1_inds] /= t_lims[t_step+1] - t_lims[t_step]
        stable_timefraction[quarter2_inds] /= t_lims[t_step+1] - max(t_lims[t_step], var.TIME_ENLARGE1)
        stable_timefraction[quarter34_inds] /= t_lims[t_step+1] - max(t_lims[t_step], var.TIME_ENLARGE2)

        # calculate the stability
        stab_per_pixel = stable_timefraction[:, 0] # get time fraction for most stable pixel
        inds_nonzero = np.where(stab_per_pixel>1e-10)
        stab_per_pixel = stab_per_pixel[inds_nonzero] # remove zeros
        if not np.any(inds_nonzero):
            stability = 1
        else:
            stability = np.mean(stab_per_pixel)
        stability_vs_time[t_step] = stability

        # save full result to pickle file
        if save_pickle:
            file_path = os.path.join(os.path.join(os.getcwd(), 'data'), 'stability_' + canvas_part.id + '_time{:06d}to{:06d}.pickle'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
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
                os.makedirs(os.path.join(os.getcwd(), 'figs', 'history_' + canvas_part.id))
            except OSError: 
                print('')
            im1 = Image.fromarray(pixels1.astype(np.uint8))
            im1_path = os.path.join(os.getcwd(), 'figs','history_' + canvas_part.id, 'MostStableColor_time{:06d}to{:06d}.png'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
            im1.save(im1_path)
            im2 = Image.fromarray(pixels2.astype(np.uint8))
            im2_path = os.path.join(os.getcwd(), 'figs','history_' + canvas_part.id, 'SecondMostStableColor_time{:06d}to{:06d}.png'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
            im2.save(im2_path)
            im3 = Image.fromarray(pixels3.astype(np.uint8))
            im3_path = os.path.join(os.getcwd(), 'figs','history_' + canvas_part.id, 'ThirdMostStableColor_time{:06d}to{:06d}.png'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
            im3.save(im3_path)
    
    return stability_vs_time

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
    stability array (and colors sorted by dominance) saved in pickle file, for each time step.
    Returns the stability averaged over all the pixels, in a list for each time range
    '''

    if t_lims[0] != 0:
        print('WARNING: this function \'stability\' is not meant to work from a lower time limit > 0 !!!')

    seconds = np.array(canvas_part.pixel_changes['seconds'])
    xcoor = np.array(canvas_part.pixel_changes['xcoor'])
    ycoor = np.array(canvas_part.pixel_changes['ycoor'])
    color = np.array(canvas_part.pixel_changes['color'])

    color_dict = json.load(open(os.path.join(canvas_part.data_path, 'ColorDict.json')))
    white = color_dict['#FFFFFF']
    current_color = np.full((canvas_part.xmax - canvas_part.xmin + 1, canvas_part.ymax - canvas_part.ymin + 1), white, dtype='int8')

    stability_vs_time = []

    for t_step in range(0, len(t_lims)-1):
        t_inds = np.where((seconds >= t_lims[t_step]) & (seconds < t_lims[t_step+1]))[0] #  & (xcoor >= xmin) & (xcoor <= xmax) & (ycoor >= ymin) & (ycoor <= ymax)

        time_spent_in_color = np.zeros((canvas_part.xmax - canvas_part.xmin + 1, canvas_part.ymax - canvas_part.ymin + 1, len(color_dict)), dtype='float64')
        last_time_changed = np.full((canvas_part.xmax - canvas_part.xmin + 1, canvas_part.ymax - canvas_part.ymin + 1), t_lims[t_step], dtype='float64')
        # neglect the time before the opening of the supplementary canvas quarters
        last_time_changed[max(1000-canvas_part.xmin, 0):, :max(1000-canvas_part.ymin, 0)] = max(t_lims[t_step], var.TIME_ENLARGE1)
        last_time_changed[:, max(1000-canvas_part.ymin, 0):] = max(t_lims[t_step], var.TIME_ENLARGE2)

        for tidx in t_inds:
            x = xcoor[tidx] - canvas_part.xmin # defines the x_ind for a given pixel change time
            y = ycoor[tidx] - canvas_part.ymin # instead we could just define the x_ind for a given pixel change time
            s = seconds[tidx]
            c = color[tidx]

            # add the time that this pixel spent in the most recent color
            time_spent_in_color[x,y, current_color[x,y]] += s - last_time_changed[x,y]

            # update the time and color of the last pixel change for this pixel
            last_time_changed[x, y] = s # should probably be renamed to prev_time_changed
            current_color[x, y] = c

        # add the time spent in the final color (from the last pixel change to the end-time
        for x in range(0, canvas_part.xmax-canvas_part.xmin+1):
            for y in range(0, canvas_part.ymax-canvas_part.ymin+1):
                time_spent_in_color[x, y, current_color[x, y]] += max(t_lims[t_step+1] - last_time_changed[x, y], 0)
        
        # get the color where pixels spent the most time
        stable_colors = np.flip(np.argsort(time_spent_in_color, axis=2), axis=2)  # sort in descending order
        stable_timefraction = np.take_along_axis(time_spent_in_color, stable_colors, axis=2)
        # normalize by the total time the canvas quarter was on
        stable_timefraction[:max(1000-canvas_part.xmin, 0), :max(1000-canvas_part.ymin, 0), :] /= t_lims[t_step+1] - t_lims[t_step]
        stable_timefraction[max(1000-canvas_part.xmin, 0):, :max(1000-canvas_part.ymin, 0), :] /= t_lims[t_step+1] - max(t_lims[t_step], var.TIME_ENLARGE1)
        stable_timefraction[:, max(1000-canvas_part.ymin, 0):, :] /= t_lims[t_step+1] - max(t_lims[t_step], var.TIME_ENLARGE2)

        # check that sum along axis=2 is t+1 - t
        '''
        sum_timefraction = np.sum(stable_timefraction, axis=2)
        for x in range(0, canvas_part.xmax-canvas_part.xmin+1):
            for y in range(0, canvas_part.ymax-canvas_part.ymin+1):
                if abs(sum_timefraction[x, y]-1) > 1e-6:
                    print("problem with time counting in pixel x,y=", x, y, "!!!: ", sum_timefraction[x, y]-1)
        '''

        # average of the stability over all pixels of the canvas_part, for this time range
        stability_pixelAverage = 0
        counter = 0
        for x,y in zip(canvas_part.coords[0], canvas_part.coords[1]):
            stab = stable_timefraction[x-canvas_part.xmin, y-canvas_part.ymin, 0]
            if stab > 1e-10: # only add if values is greater than zero #remove pixels where stab==0 (meaning they were not 'on' in this time range)
                stability_pixelAverage += stab # add the stable time fraction to the summed time fraction
                counter += 1 
        if counter > 0: 
            stability_pixelAverage /= counter
        else: # case where no pixel was active
            stability_pixelAverage = 1
        stability_vs_time.append(stability_pixelAverage)

        # save full result to pickle file
        if save_pickle:
            file_path = os.path.join(os.path.join(os.getcwd(), 'data'), 'stability_' + canvas_part.id + '_time{:06d}to{:06d}.pickle'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
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

            pixels1 = canvas_part.get_rgb(np.swapaxes(stable_colors[:,:,0], 0, 1))  
            pixels2 = canvas_part.get_rgb(np.swapaxes(stable_colors[:,:,1], 0, 1))
            pixels3 = canvas_part.get_rgb(np.swapaxes(stable_colors[:,:,2], 0, 1))
            # if second and/or third most used colors don't exist (time_spent == 0), then use the first or second most used color instead
            inds_to_change1 = np.where(stable_timefraction[:,:,1] < 1e-9)
            pixels2[inds_to_change1] = pixels1[inds_to_change1] 
            inds_to_change2 = np.where(stable_timefraction[:,:,2] < 1e-9)
            pixels3[inds_to_change2] = pixels2[inds_to_change2] # also changes to the first most used if there is no second most used color

            # save images
            try:
                os.makedirs(os.path.join(os.getcwd(), 'figs', 'history_' + canvas_part.id))
            except OSError: 
                print('')
            im1 = Image.fromarray(pixels1.astype(np.uint8))
            im1_path = os.path.join(os.getcwd(), 'figs','history_' + canvas_part.id, 'MostStableColor_time{:06d}to{:06d}.png'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
            im1.save(im1_path)
            im2 = Image.fromarray(pixels2.astype(np.uint8))
            im2_path = os.path.join(os.getcwd(), 'figs','history_' + canvas_part.id, 'SecondMostStableColor_time{:06d}to{:06d}.png'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
            im2.save(im2_path)
            im3 = Image.fromarray(pixels3.astype(np.uint8))
            im3_path = os.path.join(os.getcwd(), 'figs','history_' + canvas_part.id, 'ThirdMostStableColor_time{:06d}to{:06d}.png'.format(int(t_lims[t_step]), int(t_lims[t_step+1])))
            im3.save(im3_path)
    
    return np.asarray(stability_vs_time)


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

