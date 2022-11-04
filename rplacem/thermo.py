import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
import os


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
    for i in range(0,len(time_inds_list)):

        # get number of pixel changes that have had at least one change
        # since the start

        # get the pixel change coordinates for the interval
        pixel_changes_time_int = canvas_part.pixel_changes.iloc[time_inds_list[i],:]
        x_coord = np.array(pixel_changes_time_int['x_coord'])
        y_coord = np.array(pixel_changes_time_int['y_coord'])
        user_id = np.array(pixel_changes_time_int['user_id'])
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


def plot_compression_vs_pixel_changes(num_pixel_changes, 
                                      num_touched_pixels, 
                                      num_users,
                                      time_interval,
                                      file_size_png, 
                                      file_size_bmp):

    '''
    plot the pixel change quanities and CID (computable information density)
    '''
    time_sec = time_interval*np.linspace(1,len(num_pixel_changes),len(num_pixel_changes))
    fig_cid_vs_time = plt.figure()
    plt.plot(time_sec, file_size_png/file_size_bmp)
    plt.ylabel('Computational Information Density  (file size ratio)')
    plt.xlabel('Time (s)')
    sns.despine()

    fig_num_touched_pix_vs_time = plt.figure()
    plt.plot(time_sec, num_touched_pixels)
    plt.ylabel('Number of touched pixels')
    plt.xlabel('Time (s)')
    sns.despine()

    fig_num_pix_changes_vs_time = plt.figure()
    plt.plot(time_sec, num_pixel_changes)
    plt.ylabel('Number of Pixel Changes')
    plt.xlabel('Time (s)')
    sns.despine()

    fig_users_vs_time = plt.figure()
    plt.plot(time_sec, num_users)
    plt.ylabel('Number of Users')
    plt.xlabel('Time (s)')
    sns.despine()

    fig_cid_vs_num_pix_changes = plt.figure()
    plt.scatter(num_pixel_changes, file_size_png/file_size_bmp, s=5, alpha=0.7, c=time_sec)
    plt.xlabel('Number of Pixel Changes')
    plt.ylabel('Computational Information Density (file size ratio)')
    sns.despine()

    fig_cid_vs_num_touched_pix = plt.figure()
    plt.scatter(num_touched_pixels, file_size_png/file_size_bmp, s=5, alpha=0.7, c=time_sec)
    plt.xlabel('Number of touched Pixels')
    plt.ylabel('Computational Information Density (file size ratio)')
    sns.despine()

    fig_cid_vs_num_users = plt.figure()
    plt.scatter(num_users, file_size_png/file_size_bmp, s=5, alpha=0.7, c=time_sec)
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