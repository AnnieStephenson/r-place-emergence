import matplotlib.pyplot as plt
import seaborn as sns
import os
import rplacem.variables_rplace2022 as var

def show_canvas_part(pixels, ax=None):
    '''
    Plots 2D pixels array

    '''
    if ax == None:
        plt.figure(origin='upper')
        plt.imshow(pixels, origin='upper')
    else:
        ax.imshow(pixels, origin='upper')

def compression(file_size_bmp, file_size_png, times, out_name=''):
    '''
    plot the file size ratio over time

    parameters
    ----------
    file_size_bmp : float
        size of png image in bytes
    file_size_png : float
        size of png image in bytes
    times : 1d array of floats
        time intervals at which to plot (in seconds)
    out_name : string
        for the naming of the output saved plot
    '''

    plt.figure()
    plt.plot(times, file_size_png/file_size_bmp)
    sns.despine()
    plt.ylabel('Computable Information Density (file size ratio)')
    plt.xlabel('Time (s)')
    plt.savefig(os.path.join(var.FIGS_PATH, out_name, '_file_size_compression_ratio.png'))

def compression_vs_pixel_changes(num_pixel_changes,
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
