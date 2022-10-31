import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
import os


def calc_num_pixel_changes(canvas_part,
                           pixel_changes_all,
                           time_inds_list,
                           time_interval):
    '''
    calculate several quantities related to the rate of pixel changes 

    parameters
    ----------
    canvas_part : CanvasPart object
        The CanvasPart object for which we want to calculate the pixel change quantities
    pixel_changes_all : numpy array
        Pandas dataframe containing all the pixel change. Output of get_all_pixel_changes()
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
    num_pixel_changes_per_interval : 1d numpy array (length: number of time steps)
        The number of pixel changes in each time interval
        (Temperature)
    num_touched_pixels_per_interval : 1d numpy array (length: number of time steps)
        The number of touched pixels (only one change counted per pixel) in each time interval
    num_active_users: 1d numpy array (length: number of times steps)
        The number of active user at each time interval
    '''

    num_pixel_changes = np.zeros(len(time_inds_list))
    num_touched_pixels = np.zeros(len(time_inds_list))
    num_pixel_changes_per_interval = np.zeros(len(time_inds_list))
    num_touched_pixels_per_interval = np.zeros(len(time_inds_list))
    num_users_per_interval = np.zeros(len(time_inds_list))

    for i in range(0,len(time_inds_list)):
        # total number of pixel changes since the starting time
        num_pixel_changes[i] = len(time_inds_list[i])

        # get number of pixel changes that have had at least one change
        # since the start

        # # get the pixel change coordinates since the start
        pixel_changes_time_int = canvas_part.pixel_changes.iloc[time_inds_list[i],:]
        x_coord = np.array(pixel_changes_time_int['x_coord'])
        y_coord = np.array(pixel_changes_time_int['y_coord'])
        user_id = np.array(pixel_changes_time_int['user_id'])
        coords = np.vstack((x_coord, y_coord))

        # get rid of the duplicate pixel changes
        unique_pixel_changes = np.unique(coords, axis=1)
        num_touched_pixels[i] = unique_pixel_changes.shape[1]

        # get number of pixel changes that have had at least one change 
        # since the previous interval                    
        
        # find the intersection of time_inds_list current and time_inds_list previous step
        intersect, ind_t0, ind_t1 = np.intersect1d(time_inds_list[i], time_inds_list[i-1], return_indices=True)

        # find the time indices that do not correspond to the intersection of the two time steps
        time_inds_interval = time_inds_list[i][len(ind_t0):]

        # get the number of unique user ids
        unique_ids = np.unique(user_id[time_inds_interval])
        num_users_per_interval[i] = unique_ids.shape[0]

        # get the pixel change coordinates for those time intervals
        pixel_changes_since_time_int = canvas_part.pixel_changes.iloc[time_inds_interval,:]
        x_coord = np.array(pixel_changes_since_time_int['x_coord'])
        y_coord = np.array(pixel_changes_since_time_int['y_coord'])
        coords = np.vstack((x_coord, y_coord))

        # get rid of duplicate pixel changes
        unique_pixel_changes_int = np.unique(coords, axis=1)

        # take the length to count
        num_touched_pixels_per_interval[i] = unique_pixel_changes_int.shape[1]

        # number of pixel changes within the current time interval

        # subtract the length of the previous time index array from the current one 
        num_pixel_changes_per_interval[i] = len(time_inds_list[i]) - len(time_inds_list[i-1])

        # set the first one to zero since the previous line gives it a negative number
        num_pixel_changes_per_interval[0] = 0 

    return (num_pixel_changes, 
            num_touched_pixels, 
            num_pixel_changes_per_interval, 
            num_touched_pixels_per_interval,
            num_users_per_interval)



def plot_compression_vs_pixel_changes(num_pixel_changes, 
                                      num_touched_pixels, 
                                      num_pixel_changes_per_interval, 
                                      num_touched_pixels_per_interval, 
                                      num_users_per_interval,
                                      time_interval,
                                      file_size_png, file_size_bmp):

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

    fig_num_touched_pix_int_vs_time = plt.figure()
    plt.plot(time_sec, num_touched_pixels_per_interval)
    plt.ylabel('Number of touched pixels per Interval')
    plt.xlabel('Time (s)')
    sns.despine()

    fig_num_pix_changes_int_vs_time = plt.figure()
    plt.plot(time_sec, num_pixel_changes_per_interval)
    plt.ylabel('Number of Pixel Changes per Interval')
    #plt.ylim([0,3000])
    plt.xlabel('Time (s)')
    sns.despine()

    fig_users_int_vs_time = plt.figure()
    plt.plot(time_sec, num_users_per_interval)
    plt.ylabel('Number of Users per Interval')
    #plt.ylim([0,3000])
    plt.xlabel('Time (s)')
    sns.despine()

    fig_cid_vs_num_pix_changes = plt.figure()
    plt.plot(num_pixel_changes, file_size_png/file_size_bmp)
    plt.xlabel('Number of Pixel Changes')
    plt.ylabel('Computational Information Density (file size ratio)')
    sns.despine()

    fig_cid_vs_num_touched_pix = plt.figure()
    plt.scatter(num_touched_pixels, file_size_png/file_size_bmp, s=5, alpha=0.7, c=time_sec)
    plt.xlabel('Number of touched Pixels')
    plt.ylabel('Computational Information Density (file size ratio)')
    sns.despine()

    fig_cid_vs_num_pix_changes_per_interval = plt.figure()
    plt.scatter(num_pixel_changes_per_interval, file_size_png/file_size_bmp,s=5, alpha=0.7, c=time_sec)
    plt.ylabel('Computational Information Density (file size ratio)')
    plt.xlabel('Number of Pixel Changes per Interval')
    plt.xlim([0, 2000])
    sns.despine()

    fig_cid_vs_num_touched_pixels_per_interval = plt.figure()
    plt.scatter(num_touched_pixels_per_interval, file_size_png/file_size_bmp,s=5, alpha=0.7, c=time_sec)
    plt.ylabel('Computational Information Density (file size ratio)')
    plt.xlabel('Number of touched pixels per Interval')
    plt.xlim([0, 1000])
    sns.despine()

    fig_cid_vs_num_users_per_interval = plt.figure()
    plt.scatter(num_users_per_interval, file_size_png/file_size_bmp, s=5, alpha=0.7, c=time_sec)
    plt.ylabel('Computational Information Density (file size ratio)')
    plt.xlabel('Number of Users per Interval')
    plt.xlim([0, 1000])
    sns.despine()

    return (fig_cid_vs_time, 
            fig_num_touched_pix_vs_time, 
            fig_num_pix_changes_vs_time, 
            fig_num_pix_changes_int_vs_time, 
            fig_cid_vs_num_pix_changes,
            fig_cid_vs_num_touched_pix,
            fig_cid_vs_num_pix_changes_per_interval,
            fig_cid_vs_num_touched_pixels_per_interval,
            fig_users_int_vs_time,
            fig_cid_vs_num_users_per_interval)