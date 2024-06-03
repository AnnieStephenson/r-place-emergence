import matplotlib as mpl
mpl.use('agg') # clears memory problem with GUI backend (plt.close() not working when plt.show() wasn't run before)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import ScalarFormatter
import mpl_toolkits.axes_grid1.inset_locator as insloc
import seaborn as sns
import os, copy
import rplacem.globalvariables_peryear as vars
var = vars.var
import rplacem.utilities as util
import numpy as np
import math

def show_canvas_part(pixels, ax=None):
    '''
    Plots 2D pixels array

    '''
    if ax == None:
        plt.figure(origin='upper')
        plt.imshow(pixels, origin='upper')
    else:
        ax.imshow(pixels, origin='upper')


def show_canvas_part_over_time(pixels_vst, figsize=(3,3)):
    '''
    Plots a grid of the canvas part instantaneous images over time
    '''
    num_time_steps = len(pixels_vst)
    ncols = np.min([num_time_steps, 10])
    nrows = np.max([1, int(math.ceil(num_time_steps/10))])
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < num_time_steps:
            show_canvas_part(util.get_rgb(pixels_vst[i]), ax=ax)


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
                scientific_labels=True, alreadyhist=False,
                fontsize=14, fontsize_label=17, linecolor=[0.2, 0.2, 0.2], linewidth=1):
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
        plt.stairs(data, bins, lw=linewidth, facecolor='b', edgecolor=linecolor)
    elif len(bins) == 1:
        if xrange==[]:
            res = [np.min(data), np.max(data)]
            spread = res[1] - res[0]
            xrange = [res[0] - spread*2/bins, res[1] + spread*2/bins]

        plt.hist(data, bins=bins[0], range=xrange, lw=linewidth, histtype='step', facecolor='b', edgecolor=linecolor)
    else:
        plt.hist(data, bins=bins, lw=linewidth, histtype='step', facecolor='b', edgecolor=linecolor)

    sns.despine()
    plt.ticklabel_format(style=('scientific' if scientific_labels else 'plain'),
                         scilimits=([0,0] if scientific_labels else [-10,10]) )
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.xaxis.offsetText.set_fontsize(fontsize)
    ax.yaxis.offsetText.set_fontsize(fontsize)
    plt.xlabel(xlab, fontsize=fontsize_label)
    plt.ylabel(ylab, fontsize=fontsize_label)

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

def draw_2dmap(h2d, 
               logz=True,
               zmax=1500,
               clabel='',
               outfile='',
               cmap_log='cividis',
               cmap = 'inferno',
               colorbar_orient='horizontal',
               cbar_ticks_pos = 'top',
               fontsize_ticks = 20):
    '''
    plots and saves a 2D histogram in the style of a heatmap, including a colormap.
    Mostly designed for the whole 2000x2000 canvas

    parameters
    ----------
    h2d : histogram as output by np.histogram2d (though histogram must be already transposed)
    logz: log for the colormap axis
    '''
    # DO NOT CHANGE THINGS RELATED TO RESOLUTION OR SIZES: everything is adjusted to have perfect resolution for 2000x2000 canvas
    imageres = 2000 #number of pixels of the image to be shown (for perfect resolution)

    fig = plt.figure()
    addspace = 0.75
    fig.subplots_adjust(left=0,right=1,bottom=0,top=addspace)
    fig.set_size_inches(10*(1.5 if (var.year==2023) else 1), 10/addspace, forward=True)
    fig.tight_layout(pad=0.)

    if not logz:
        h2d = np.ma.masked_where(h2d == 0, h2d)
        #lin_cmap.set_bad(color='white') # so that the bins with content 0 are not drawn
    map = plt.imshow(h2d, interpolation='none', origin='lower', aspect=1,
                     cmap=(cmap_log if logz else cmap), norm=(colors.LogNorm(1,zmax) if logz else colors.Normalize(vmin=0, vmax=zmax)))

    plt.axis('off')
    ax = plt.gca()
    axins = insloc.inset_axes(ax, width="80%", height="3.5%", loc="upper center",
                              bbox_to_anchor=(0., 0.043, 1, 1), bbox_transform=ax.transAxes, borderpad=0)

    cbar = fig.colorbar(map, cax=axins, orientation=colorbar_orient)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.xaxis.set_ticks_position(cbar_ticks_pos)
    cbar.ax.xaxis.set_label_position(cbar_ticks_pos)
    cbar.set_label(clabel, fontsize=fontsize_ticks, labelpad=10)

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
    colorder = np.array([31,22,18,12,5,17,20,4,11,2,1,9,10,13,0,3,6,7,8,16,30,29,28,27,25,19,15,14,21,23,24,26]) if var.year == 2022 else \
               np.array([3,0,8,27,4,18,16,29,19,23,30,13,12,9,28,22,20,11,17,15,26,1,31,5,2,6,7,24,14,25,21,10])
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

def draw_1d(xdata,
            ydata,
            xlab='',
            ylab='',
            xlog=False,
            ylog=False,
            xmin=None,
            ymin=None,
            ymax=None,
            save='',
            hline=None,
            vline=None
            ):
    '''
    Draw 1d plot from full x and y information
    '''
    plt.figure()
    plt.plot(xdata, ydata)
    sns.despine()
    plt.ylabel(ylab)
    plt.xlabel(xlab)

    xm = min(xdata)
    if xm == 0 and xlog:
        xm = 1e-3
    if ymin is None:
        ym = min(ydata)
        if ym == 0 and ylog:
            ym = 1e-5
    else:
        ym = ymin

    if ymax is None:
        yM = (1.6 if ylog else 1.1) * max(ydata)
    else:
        yM = ymax
    if yM <= ym:
        yM = ym + 0.01
    plt.ylim([ym , yM])
    plt.xlim([ (xm if xmin == None else xmin), max(xdata)])

    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')

    if hline != None:
        plt.hlines(y = hline, xmin=xm, xmax=max(xdata), colors = 'black', linestyle='dashed')
    if vline != None:
        plt.vlines(x = vline, ymin=ym, ymax=ymax, colors = 'black', linestyle='dashed')

    if save != '':
        plt.savefig(save, dpi=250, bbox_inches='tight')
        plt.close()

def cpstat_tseries(cpstat, nrows=8, ncols=2, figsize=(5,10), fontsize=5, save=True):

    itmin = np.argmax(cpstat.t_lims >= cpstat.tmin)

    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=figsize)

    t_series_vars = [[cpstat.frac_pixdiff_inst_vs_stable_norm, 0, None],
                     [cpstat.frac_pixdiff_inst_vs_swref, 0, None],
                     [cpstat.instability_norm[0], 0, None],
                     [cpstat.entropy, 0, None],
                     [cpstat.frac_redundant_color_changes, 0, None],
                     [cpstat.n_changes_norm, 0, None],
                     [cpstat.frac_attack_changes, 0, 1],
                     [cpstat.frac_users_new_vs_sw, 0, 1],
                     [cpstat.runnerup_timeratio[0], 0, None],
                     [cpstat.n_used_colors[0], 1, None],
                     #[cpstat.frac_users_new_vs_previoustime, 0, 1],
                     [cpstat.n_users_sw_norm, 0, None],
                     [cpstat.changes_per_user_sw, 0.9, None],
                     [cpstat.fractal_dim_weighted, None, None],
                     #[cpstat.frac_cooldowncheat_changes, 0, None],
                     #[cpstat.frac_bothattdef_users, 0, None],
                     [cpstat.returntime[3], 0, None],
                     [cpstat.returntime[0], 0, None],
                     [cpstat.cumul_attack_timefrac, 0, None],
                     [cpstat.variance_multinom, 0, None],
                     [cpstat.variance_from_frac_pixdiff_inst, 0, 0.001],
                     [cpstat.variance2, 1, None],
                     [cpstat.autocorr_subdom, 0, None],
                     [cpstat.autocorr_multinom, 0, None],
                     [cpstat.returnrate, 0, 1],
                     #[cpstat.n_users_norm, 0, None],
                     #[cpstat.frac_attackonly_users, 0, 1]
                    ]

    for i, ax in enumerate(axes.T.flat):
        #print(i)
        #print(t_series_vars[i][0].desc_long)
        #print(t_series_vars[i][0].val[0:50])
        ax.plot(cpstat.t_lims[itmin:], t_series_vars[i][0].val[itmin:])
        ax.patch.set_alpha(0)

        ax.set_xlim([cpstat.t_lims[itmin], cpstat.t_lims[-1]])
        ax.tick_params(axis='x', direction='in')

        reject_end = int(t_series_vars[0][0].n_pts * 6./300.) # reject ending white period, and the very beginning
        ym = min(t_series_vars[i][0].val[4:-reject_end])
        yM = max(t_series_vars[i][0].val[4:-reject_end])
        ymin = ym - 0.1*(yM-ym) if t_series_vars[i][1] is None else t_series_vars[i][1]
        ymax = yM + 0.1*(yM-ym) if t_series_vars[i][2] is None else t_series_vars[i][2]
        ax.set_ylim([ymin + 1e-6, ymax - 1e-6])
        ax.tick_params(axis='y', which='major', labelsize=8)

        ax.set_title(t_series_vars[i][0].label, fontsize=fontsize, y=1.0, pad=-15)
        sns.despine()

    plt.subplots_adjust(hspace=0.0)
    fig.text(0.52, 0.08, 'Time [s]', ha='center')

    if save:
        plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'all_time_series.pdf'), dpi=250, bbox_inches='tight')

def draw_2dplot(x, y, z, 
                xlab='', ylab='', zlab='',
                ymax=None,
                logz=False, zmin=None, zmax=None,
                force_scientific=False,
                outname=''):
    '''
    Draw 2d (pcolormesh) plot from full x, y, z information
    '''
    plt.figure()
    zm = (1e-3 if logz else 0) if zmin == None else zmin
    zM = 1.1 * np.amax(z) if zmax == None else zmax
    plt.pcolormesh(x, y, np.transpose(z),
                   cmap=('cividis' if logz else 'inferno'), 
                   norm=colors.LogNorm(vmin=zm, vmax=zM) if logz else None,
                   vmin=None if logz else zm, vmax=None if logz else zM,
                   shading='nearest'
                   )
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.ylim([y[0], y[-1] if ymax == None else ymax])
    if force_scientific:
        plt.ticklabel_format(style='scientific')
    plt.colorbar(label=zlab)
    if outname != '':
        plt.savefig(os.path.join(var.FIGS_PATH, outname), bbox_inches='tight')
        plt.close()
