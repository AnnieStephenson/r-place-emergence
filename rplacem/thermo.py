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
            x = xcoor[tidx] - canvas_part.xmin
            y = ycoor[tidx] - canvas_part.ymin
            s = seconds[tidx]
            c = color[tidx]

            # add the time that this pixel spent in the most recent color
            time_spent_in_color[x, y, current_color[x, y]] += s - last_time_changed[x, y]

            # update the time and color of the last pixel change for this pixel
            last_time_changed[x, y] = s
            current_color[x, y] = c

        # add the time spent in the final color (from the last pixel change to the end-time)
        for x in range(0, canvas_part.xmax-canvas_part.xmin+1):
            for y in range(0, canvas_part.ymax-canvas_part.ymin+1):
                time_spent_in_color[x, y, current_color[x, y]] += max(t_lims[t_step+1] - last_time_changed[x, y], 0)

        # get the color where pixels spent the most time
        stable_colors = np.flip(np.argsort(time_spent_in_color, axis=2), axis=2)  # sort in descending order
        stable_timefraction = np.take_along_axis(time_spent_in_color, stable_colors, axis=2)
        # normalize by the total time the canvas quarter was on
        stable_timefraction[:max(1000-canvas_part.xmin, 0), :max(1000-canvas_part.ymin, 0), :] /= t_lims[t_step+1] - t_lims[t_step]
        stable_timefraction[max(1000-canvas_part.xmin, 0):, :max(1000-canvas_part.ymin, 0), :] /= t_lims[t_step+1] - max(t_lims[t_step], var.TIME_ENLARGE1)
        stable_timefraction[:,                    max(1000-canvas_part.ymin, 0):, :] /= t_lims[t_step+1] - max(t_lims[t_step], var.TIME_ENLARGE2)

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
            if stab > 1e-10: #remove pixels where stab==0 (meaning they were not 'on' in this time range)
                stability_pixelAverage += stab
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
    print(file_size_png.shape)
    print(times.shape)
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