import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mpl_toolkits.axes_grid1.inset_locator as insloc
import seaborn as sns
import os, copy
import rplacem.variables_rplace2022 as var
import rplacem.utilities as util
import numpy as np


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

def draw_1dhist(data, xrange=[], bins=[100],
                xlab='', ylab='',
                xlog=False, ylog=False, x0log=0,
                outfile='',
                scientific_labels=True, alreadyhist=False):
    '''
    plot the histogram of the given 1d array and saves it

    parameters
    ----------
    data: numpy 1d array that will be plotted
    bins: bin limits. If size=1, it is interpreted as a number of bins over the range xrange
    x0log: sets the minimum x when xlog=True
    alreadyhist: if true, the input 'data' should be the output of a np.histogram() call
    '''

    fig, ax = plt.subplots()

    if alreadyhist:
        plt.stairs(data, bins, lw=1, facecolor='b')
    elif len(bins) == 1:
        if xrange==[]:
            res = [np.min(data), np.max(data)]
            spread = res[1] - res[0]
            xrange = [res[0] - spread*2/bins, res[1] + spread*2/bins]

        plt.hist(data, bins=bins[0], range=xrange, lw=1, histtype='step', facecolor='b')
    else:
        plt.hist(data, bins=bins, lw=1, histtype='step', facecolor='b')

    sns.despine()
    plt.ticklabel_format(style=('scientific' if scientific_labels else 'plain'),
                         scilimits=([0,0] if scientific_labels else [-10,10]) )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.xaxis.offsetText.set_fontsize(14)
    ax.yaxis.offsetText.set_fontsize(14)
    plt.xlabel(xlab, fontsize=17)
    plt.ylabel(ylab, fontsize=17)

    xlimits = xrange if len(bins) == 1 else [bins[0],bins[-1]]
    if xlog or x0log != 0:
        plt.xscale('log')
        if xlimits[0] <= 0:
            xlimits[0] = x0log
    plt.xlim(xlimits)
    if ylog:
        plt.yscale('log')

    if outfile != '':
        print('save figure in', os.path.join(var.FIGS_PATH, outfile))
        plt.savefig(os.path.join(var.FIGS_PATH, outfile), bbox_inches='tight')

def draw_2dmap(h2d, xedges, yedges,
               logz=True,
               zmax=1500,
               clabel='',
               outfile=''):
    '''
    plots and saves a 2D histogram in the style of a heatmap, including a colormap

    parameters
    ----------
    h2d, xedges, yedges: histogram as output by np.histogram2d (though histogram must be already transposed)
    logz: log for the colormap axis
    '''
    # DO NOT CHANGE THINGS RELATED TO RESOLUTION OR SIZES: everything is adjusted to have perfect resolution for 2000x2000 canvas
    imageres = 2000 #number of pixels of the image to be shown (for perfect resolution)

    fig = plt.figure()
    addspace = 0.75
    fig.subplots_adjust(left=0,right=1,bottom=0,top=addspace)
    fig.set_size_inches(10,10/addspace, forward=True)
    fig.tight_layout(pad=0.)

    if not logz:
        h2d = np.ma.masked_where(h2d == 0, h2d)
        #lin_cmap.set_bad(color='white') # so that the bins with content 0 are not drawn
    map = plt.imshow(h2d, interpolation='none', origin='lower', aspect=1,
                     cmap=('inferno' if logz else 'cividis'), norm=(colors.LogNorm(1,zmax) if logz else colors.Normalize(vmin=0, vmax=zmax)))

    plt.axis('off')
    ax = plt.gca()
    axins = insloc.inset_axes(ax, width="80%", height="3.5%", loc="upper center",
                              bbox_to_anchor=(0., 0.043, 1, 1), bbox_transform=ax.transAxes, borderpad=0)

    cbar = fig.colorbar(map, cax=axins, orientation="horizontal")
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label(clabel, fontsize=24, labelpad=10)

    if outfile != '':
        print('save figure in ',os.path.join(var.FIGS_PATH, outfile))
        plt.savefig(os.path.join(var.FIGS_PATH, outfile), dpi=imageres/10, bbox_inches='tight', pad_inches = 0)
        plt.close()


def draw_colorhist(data,
                   ylab='# pixel changes',
                   ylog=False,
                   outfile=''):
    '''
    plots the histogram of colors of pixels or pixel changes, from data of color indices
    '''

    fig, ax = plt.subplots()

    xpos = np.arange(var.NUM_COLORS)
    colorder = np.array([31,22,18,12,5,17,20,4,11,2,1,9,10,13,0,3,6,7,8,16,30,29,28,27,25,19,15,14,21,23,24,26])
    colorder_inv = np.argsort(colorder)
    col = [var.IDX_TO_COLOR[str(colorder[i])] for i in xpos]
    h,b, patches = plt.hist(colorder_inv[data], bins=list(xpos)+[var.NUM_COLORS], edgecolor='black')
    for i in xpos:
        patches[i].set_facecolor(col[i])

    # y axis
    plt.ylabel(ylab, fontsize=17)
    ax.yaxis.offsetText.set_fontsize(14)
    plt.yticks(fontsize=14)
    if ylog:
        plt.yscale('log')

    # x axis
    plt.xticks(rotation=90, ha='left')
    ax.set_xticks(np.array(xpos))
    ax.set_xticklabels(col)
    plt.xlim([0,var.NUM_COLORS])

    sns.despine()
    if outfile != '':
        print('save figure in', os.path.join(var.FIGS_PATH, outfile))
        plt.savefig(os.path.join(var.FIGS_PATH, outfile), bbox_inches='tight')