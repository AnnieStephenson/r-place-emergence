import numpy as np
import os, sys
import rplacem.globalvariables_peryear as vars
var = vars.var
import numpy as np
import rplacem.canvas_part as cp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rplacem.plot_utilities as plot
import rplacem.compute_variables as comp
import seaborn as sns
import rplacem.utilities as util
import gc
import math
import pickle, json

def timeslices_ind(pixchanges, tbins):
    '''
    Returns the array of indices that separate pixchanges 
    into its elements belonging to each input time bin
    '''
    res = np.empty(len(tbins), dtype=np.int_)
    res[0] = 0
    for i in range(1,len(tbins)):
        res[i] = len(pixchanges) if i == len(tbins)-1 else np.argmax(pixchanges['seconds'][res[i-1]:] > tbins[i]) + res[i-1]
    return res

def plot_heat_map(x, y, clabel='', zmax=None, logz=True, cmap='inferno', cmap_log='cividis', outname=''):
    heat, _, _ = np.histogram2d(x, (var.CANVAS_MINMAX[-1,1,1]+var.CANVAS_MINMAX[-1,1,0])-y, 
                                    bins=[range(var.CANVAS_MINMAX[-1,0,0], var.CANVAS_MINMAX[-1,0,1]+2), 
                                          range(var.CANVAS_MINMAX[-1,1,0], var.CANVAS_MINMAX[-1,1,1]+2)])
    heat = heat.T
    if zmax == None:
        zmax = np.max(heat)

    plot.draw_2dmap(heat,
                    clabel=clabel, zmax=zmax, logz=logz, cmap=cmap, cmap_log=cmap_log,
                    outfile=outname)
    return heat

# Grab full dataset
pixel_changes_all = util.get_all_pixel_changes()
# N CHANGES
print(np.max(pixel_changes_all['seconds']))
# N USERS
print(np.max(pixel_changes_all['user']))
# N CHANGES
print(len(pixel_changes_all))





run_timedep = False
if run_timedep:
    ############ TIMES OF (DIS)APPEARANCE OF COLORS
    used_colors = []
    for p in pixel_changes_all:
        c = p['color']
        if c not in used_colors:
            used_colors.append(c)
            print(p['seconds'], 'is the time of first appeareance of color', var.IDX_TO_COLOR[str(c)])

    for c in range(32):
        print('min and max time of color', var.IDX_TO_COLOR[str(c)], 'is', 
            min(pixel_changes_all['seconds'][np.where(pixel_changes_all['color'] == c)]), 
            max(pixel_changes_all['seconds'][np.where(pixel_changes_all['color'] == c)])
            )
        
    ############ TIMES OF ENLARGMENT OF CANVAS
    print('time of enlargment 1 = ', np.min(pixel_changes_all['seconds'][np.where(pixel_changes_all['xcoor'] > 499)]))
    print('time of enlargment 2 = ', np.min(pixel_changes_all['seconds'][np.where(pixel_changes_all['xcoor'] < -500)]))
    print('time of enlargment 4 = ', np.min(pixel_changes_all['seconds'][np.where(pixel_changes_all['ycoor'] < -500)]))
    print('time of enlargment 3 = ', np.min(pixel_changes_all['seconds'][np.where(pixel_changes_all['ycoor'] > 499)]))
    print('time of enlargment 5 = ', np.min(pixel_changes_all['seconds'][np.where(pixel_changes_all['xcoor'] < -1000)]))
    print('time of enlargment 6 = ', np.min(pixel_changes_all['seconds'][np.where(pixel_changes_all['xcoor'] > 999)]))

    ############ NUM PIXEL CHANGES VS TIME
    t_interval = 300
    nbins = math.ceil(var.TIME_TOTAL/t_interval)
    plot.draw_1dhist(pixel_changes_all['seconds'][ np.where(pixel_changes_all['moderator'] == 0) ], 
                    xrange=[0,nbins*t_interval], bins=[nbins], 
                    xlab='time [s]',
                    ylab='# of pixel changes / 5 min',
                    outfile='TimeOfPixelChanges_noModerator.pdf',
                    linecolor='blue')

    plot.draw_1dhist(pixel_changes_all['seconds'][ np.where(pixel_changes_all['moderator'] == 1) ], 
                    xrange=[0,nbins*t_interval], bins=[nbins], 
                    xlab='time [s]',
                    ylab='# of pixel changes / 5 min',
                    outfile='TimeOfPixelChanges_moderatorEvents.pdf',
                    linecolor='blue')
    print('total number of moderator changes = ', np.count_nonzero(pixel_changes_all['moderator']))

    ############ PRINT USER TAGS OF MODERATORS
    user_idx2tag = json.load(open(os.path.join(var.DATA_PATH, 'userIDsFromIdx.json')))
    #print([user_idx2tag[str(u)] for u in np.unique(pixel_changes_all['user'][pixel_changes_all['moderator'] == 1])])
    print(len(np.unique(pixel_changes_all['user'][pixel_changes_all['moderator'] == 1])))

    plot.draw_1dhist(pixel_changes_all['seconds'], 
                    xrange=[0,nbins*t_interval], bins=[nbins], 
                    xlab='time [s]',
                    ylab='# of pixel changes / 5 min',
                    outfile='TimeOfPixelChanges.pdf',
                    linecolor='blue')

    ############ NUM ACTIVE USERS VS TIME
    t_interval = 10800
    nbins = math.ceil(var.TIME_TOTAL/t_interval)
    bins = np.arange(0,t_interval*(nbins+1),t_interval)
    tindices = timeslices_ind(pixel_changes_all, bins)
    users_tsliced = np.split(pixel_changes_all['user'], tindices[1:])
    nusers_pertbin = np.zeros(nbins)
    for i in range(0,nbins):
        nusers_pertbin[i] = len(np.unique(users_tsliced[i]))
    plot.draw_1dhist(nusers_pertbin, 
                    bins=bins,
                    xlab='time [s]',
                    ylab='# active users / 3h',
                    outfile='ActiveUsersNumber_VsTime.pdf',
                    alreadyhist=True,
                    linecolor='blue')





       
run_heatmaps = False
run_numpixels = False
if run_heatmaps or run_numpixels:
    ############ FULL HEAT MAP
    heat = plot_heat_map(pixel_changes_all['xcoor'], pixel_changes_all['ycoor'], 
                        clabel='# of pixel changes', zmax=1500,
                        outname='HeatMap.png')

if run_heatmaps:
    ############ HEAT MAP OF MODERATOR EVENTS
    plot_heat_map(pixel_changes_all['xcoor'][pixel_changes_all['moderator'] == 1], pixel_changes_all['ycoor'][pixel_changes_all['moderator'] == 1], 
                clabel='# of pixel changes from moderators',
                logz=(var.year==2023), zmax=(1000 if var.year==2023 else None),
                outname='HeatMap_moderatorEvents.png')

    ############ TIME DEPENDENT HEAT MAP
    t_interval = 900 # 15min
    nbins = math.ceil(var.TIME_TOTAL/t_interval)
    bins = np.arange(0,t_interval*(nbins+1),t_interval)
    tindices = timeslices_ind(pixel_changes_all, bins)
    xcoor_tsliced = np.split(pixel_changes_all['xcoor'], tindices[1:])
    ycoor_tsliced = np.split(pixel_changes_all['ycoor'], tindices[1:])
    for i in range(0,nbins):
        plot_heat_map(xcoor_tsliced[i], ycoor_tsliced[i],
                    clabel=' # of pixel changes / 15min',
                    cmap='jet', logz=False, zmax=8,
                    outname=os.path.join('timeDepHeatMap','HeatMap_time{:06d}to{:06d}.png'.format(int(i*t_interval), int(min(var.TIME_TOTAL, (i+1)*t_interval)))))

    util.save_movie(os.path.join(var.FIGS_PATH,'timeDepHeatMap'),fps=4)

if run_numpixels:
    ############ NUM CHANGES PER PIXEL
    numchanges = heat.flatten()
    bins = np.concatenate((np.array([0.]), 
                                        np.arange(1,100,1), 
                                        np.arange(100,800,10), 
                                        np.arange(800,3500,100), 
                                        np.arange(3500,10000,500), 
                                        np.arange(10000,109000,3000)
                                        ))
    binwidth = np.diff(bins)
    binwidth[0] = 1 # first bin is special
    numchanges_hist,_ = np.histogram(numchanges, bins)
    numchanges_hist = np.divide(numchanges_hist, binwidth)

    plot.draw_1dhist(numchanges_hist, 
                    bins=bins,
                    xlog=True,
                    ylog=True, 
                    xlab='# changes',
                    ylab='# pixels / bin width',
                    x0log=0.5,
                    alreadyhist=True,
                    linecolor='b',
                    outfile='ChangesPerPixel.pdf')
    print('mean and median of # changes per pixel = ', np.mean(numchanges), np.median(numchanges))

    ############ NUM PIXEL CHANGES PER SINGLE USER
    nmaxusers = max(pixel_changes_all['user'])
    _,perusercount = np.unique(pixel_changes_all['user'][np.where(pixel_changes_all['moderator'] == 0)],
                            return_counts=True)
    plot.draw_1dhist(perusercount, 
                    bins=range(1, (1001 if var.year == 2022 else 1801)),
                    xlog=True,
                    ylog=True, 
                    scientific_labels=False,
                    xlab='# pixel changes / user',
                    ylab='# users',
                    outfile='PixelChangesPerUser_noModerator.pdf',
                    linecolor='blue')
    print('mean and median of #pixel changes of single users = ', np.mean(perusercount), np.median(perusercount))

    ############ NUM PIXEL CHANGES PER COLOR
    plot.draw_colorhist(pixel_changes_all['color'], outfile='color_distribution_pixelchanges.pdf', ylog=False)






run_timediffs = True
run_timediffs_2d = True
run_spacecorrelations = True

if run_timediffs or run_timediffs_2d or run_spacecorrelations:
    ############ CALCULATE TIME DIFFERENCES BETWEEN PIXEL CHANGES FROM SAME USER
    # correlation between user times of pixel changes
    #pixel_changes_all = pixel_changes_all[0:5000000]
    pixel_changes_all = pixel_changes_all[np.where(np.logical_not(pixel_changes_all['moderator']))] # remove moderator events
    print('start sorting pixel changes by user')
    user_sorting = pixel_changes_all.argsort(order=['user','seconds']) # sorting vs user index, then vs time
    print('done sorting')
    pixel_changes_all = pixel_changes_all[user_sorting] # pixel_changes_all is now sorted
    #print('pixel_changes_all = ', pixel_changes_all)

    # remove the first index of each new user
    kept_idx = np.where(np.diff(pixel_changes_all['user']) == 0)[0]
    # get the time difference between pixel changes (n+1) and n for same user
    users = pixel_changes_all['user'][kept_idx]
    timedif = np.diff(pixel_changes_all['seconds'])[kept_idx]
    # give contiguous indices to kept users
    compress_users = np.hstack((np.hstack((0,  1 + np.where(np.diff(users) > 0)[0]))  , [len(users)-1, len(users)]))
    for i in range(0,len(compress_users)-1):
        users[compress_users[i]:compress_users[i+1]] = i
    users[-1] -= 1

if run_timediffs:
    ############# PLOTTING TIME DIFFERENCES
    # scatter plot
    print('scatter plot of time differences')
    # first shuffle the users, because current order comes from files containing 'correlated' users
    shuffler = np.arange(0, len(users))
    np.random.shuffle(shuffler)
    # now plot
    fig = plt.figure()
    plt.scatter(timedif, shuffler, s=0.03)
    plt.xlabel('time difference between two pixel changes of same user')
    plt.ylabel('user index')
    plt.savefig(os.path.join(var.FIGS_PATH, 'Time_difference_between_two_changes_same_user_2D.png'), bbox_inches='tight')

    # histogram of time differences between pixel changes
    print('histogram of time differences')
    fig2 = plt.figure()
    n, bins, patches = plt.hist(timedif, 2000, range=[0,var.TIME_TOTAL], facecolor='blue', alpha=0.5, align='mid')
    print('number of pixel changes done under 5 minutes = ', np.sum(n[0:2]), 'among', len(timedif), 'subsequent same-user pairs of changes')
    sns.despine()
    plt.xlabel('time interval between two pixel changes of same user')
    plt.ylabel('number of pixel changes')
    plt.xlim([150, 3e5])
    plt.ylim([1, 2e8])
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'Time_difference_between_two_changes_same_user.png'), bbox_inches='tight')
    plt.close('all')

    # same but zoomed on small times
    fig3 = plt.figure()
    n, bins, patches = plt.hist(timedif, 50, range=[0,600], facecolor='blue', alpha=0.5, align='mid')
    sns.despine()
    plt.xlabel('time interval between two pixel changes of same user')
    plt.ylabel('number of pixel changes')
    plt.xlim([0,600])
    plt.ylim([1e2, 3e7])
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'Time_difference_between_two_changes_same_user_zoomed.png'), bbox_inches='tight')

    print('number of pixel changes done in <0.4 seconds = ',len(timedif[np.where(timedif<0.4)]))

    # plot mean time interval between changes of each user
    print('histogram of mean time interval per user')
    mean_timedif_of_users = np.zeros((len(compress_users)-3))
    for i in range(0,len(compress_users)-3):
        mean_timedif_of_users[i] = np.mean(timedif[compress_users[i]:compress_users[i+1]])

    fig4 = plt.figure()
    n, bins, patches = plt.hist(mean_timedif_of_users, 2000, range=[0,var.TIME_TOTAL], facecolor='blue', alpha=0.5, align='mid')
    sns.despine()
    plt.xlabel('mean time interval between two pixel changes of same user')
    plt.ylabel('number of users')
    plt.xlim([150,var.TIME_TOTAL])
    plt.ylim([1., 2.5e6])
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'Mean_time_difference_between_two_changes_per_user.png'), bbox_inches='tight')
    plt.close('all')

    # plot min time interval between changes of each user
    print('histogram of min time interval per user')
    min_timedif_of_users = np.zeros((len(compress_users)-3))
    num_cheat_users = 0
    num_cheat_per_user = []
    for i in range(0,len(compress_users)-3):
        min_timedif_of_users[i] = np.min(timedif[compress_users[i]:compress_users[i+1]])
        if min_timedif_of_users[i] < 200:
            print('user #', compress_users[i], ', mean time diff =', np.mean(timedif[compress_users[i]:compress_users[i+1]]), 
                  ', number of pixel changes =', 1+compress_users[i+1]-compress_users[i], 
                  'of which cheated =', np.count_nonzero(timedif[compress_users[i]:compress_users[i+1]] < 200))
            num_cheat_users += 1
            num_cheat_per_user.append(np.count_nonzero(timedif[compress_users[i]:compress_users[i+1]] < 200))
    print('number of users having cheated =', num_cheat_users)

    # plot heat map of pixels that have been cheated
    plot_heat_map(pixel_changes_all['xcoor'][kept_idx][timedif < 300], pixel_changes_all['ycoor'][kept_idx][timedif < 300], 
                clabel='# of cheated pixel changes',
                logz=False, zmax=None,
                outname='HeatMap_cheatedCooldown.png')


    fig9 = plt.figure()
    n, bins, patches = plt.hist(num_cheat_per_user, 60, range=[0,60], facecolor='blue', alpha=0.5, align='mid')
    sns.despine()
    plt.xlabel('number of cheated pixel changes per user')
    plt.ylabel('number of users')
    plt.xlim([0,60])
    plt.ylim([0.8,3000])
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'Number_of_cheated_pixels_per_user.png'), bbox_inches='tight')

    fig4 = plt.figure()
    n, bins, patches = plt.hist(min_timedif_of_users, 2000, range=[0,var.TIME_TOTAL], facecolor='blue', alpha=0.5, align='mid')
    sns.despine()
    plt.xlabel('minimum time interval between two pixel changes of same user')
    plt.ylabel('number of users')
    plt.xlim([150,var.TIME_TOTAL])
    plt.ylim([1., 6e6])
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'Min_time_difference_between_two_changes_per_user.png'), bbox_inches='tight')

    # same but zoomed on zero
    fig7 = plt.figure()
    n, bins, patches = plt.hist(min_timedif_of_users, 200, range=[0,6000], facecolor='blue', alpha=0.5, align='mid')
    sns.despine()
    plt.xlabel('minimum time interval between two pixel changes of same user')
    plt.ylabel('number of users')
    plt.xlim([0, 2000])
    plt.ylim([1., 6e6])
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'Min_time_difference_between_two_changes_per_user_zoomed.png'), bbox_inches='tight')

    # plot median time interval between changes of each user
    print('histogram of median time interval per user')
    median_timedif_of_users = np.zeros((len(compress_users)-3))
    for i in range(0,len(compress_users)-3):
        median_timedif_of_users[i] = np.median(timedif[compress_users[i]:compress_users[i+1]])

    fig5 = plt.figure()
    n, bins, patches = plt.hist(median_timedif_of_users, 2000, range=[0,var.TIME_TOTAL], facecolor='blue', alpha=0.5, align='mid')
    sns.despine()
    plt.xlabel('median time interval between two pixel changes of same user')
    plt.ylabel('number of users')
    plt.xlim([150,var.TIME_TOTAL])
    plt.ylim([0.9, 2.5e6])
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'Median_time_difference_between_two_changes_per_user.png'), bbox_inches='tight')

    # plot std deviation of time intervals between changes of each user
    print('histogram of std deviation of time interval per user')
    std_timedif_of_users = np.zeros((len(compress_users)-3))
    for i in range(0,len(compress_users)-3):
        std_timedif_of_users[i] = np.std(timedif[compress_users[i]:compress_users[i+1]])

    fig6 = plt.figure()
    n, bins, patches = plt.hist(std_timedif_of_users, 2000, range=[0,var.TIME_TOTAL], facecolor='blue', alpha=0.5, align='mid')
    sns.despine()
    plt.xlabel('std deviation of time intervals between two pixel changes of same user')
    plt.ylabel('number of users')
    plt.xlim([0,150000])
    plt.ylim([0.7, 8e6])
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'StandardDev_time_difference_between_two_changes_per_user.png'), bbox_inches='tight')
    plt.close('all')

if run_timediffs_2d:
    # 2D histogram of pixel time vs time difference 
    print('Doing 2D histogram of pixel time vs time difference')
    f8 = plt.figure()
    matrix, xedge, yedge = np.histogram2d(pixel_changes_all['seconds'][kept_idx], timedif, bins=(30, 50), range=[[0,var.TIME_TOTAL],[0,1500]])
    matrix2 = np.array(matrix, copy=True)
    matrix = matrix.swapaxes(0,1)
    plt.pcolormesh(xedge, yedge, matrix, cmap=plt.cm.jet, norm=colors.LogNorm(vmin=5, vmax=3e6))
    plt.xlabel('pixel change time')
    plt.ylabel('time interval between two pixel changes of same user')
    plt.colorbar(label='number of pixel changes')
    plt.savefig(os.path.join(var.FIGS_PATH, 'Time_vs_timedifference_pixelchanges.png'), bbox_inches='tight')

    # same but normalize by sum of column contents
    for i in range(0, matrix2.shape[0]):
        norm_col = matrix2.sum(axis=1)[i]
        if norm_col != 0:
            matrix2[i] /= matrix2.sum(axis=1)[i]
    matrix2 = matrix2.swapaxes(0,1)
    f10 = plt.figure()
    plt.pcolormesh(xedge, yedge, matrix2, cmap=plt.cm.jet, norm=colors.LogNorm(vmin=8e-7, vmax=1))
    plt.xlabel('pixel change time')
    plt.ylabel('time interval between two pixel changes of same user')
    plt.colorbar(label='number of pixel changes (norm. by time interval)')
    plt.savefig(os.path.join(var.FIGS_PATH, 'Time_vs_timedifference_pixelchanges_normalized.png'), bbox_inches='tight')






if run_spacecorrelations:

    ############# COMPUTE SPACE CORRELATIONS OF PIXEL CHANGES FROM SAME USER
    print('compute space correlations')
    space_corr = []
    nchanges = [] # number of pixel changes per user
    xcoor = pixel_changes_all['xcoor']
    ycoor = pixel_changes_all['ycoor']
    users_all = pixel_changes_all['user']
    singleuser_limits = np.hstack((np.hstack((0,  1 + np.where(np.diff(users_all) > 0)[0])) , len(users_all)))

    for i in range(0, len(singleuser_limits)-1):
        n = singleuser_limits[i+1] - singleuser_limits[i]
        if n < 2: # remove users having a single pixel change
            continue
        x = xcoor[singleuser_limits[i]:singleuser_limits[i+1]]
        y = ycoor[singleuser_limits[i]:singleuser_limits[i+1]]
        xmean = np.mean(x)
        ymean = np.mean(y)
        distances = np.sqrt( (x-xmean)**2 + (y-ymean)**2 )
        space_corr.append( np.mean(distances) ) # rms was tested and gives similar result: np.sqrt(np.mean(distances**2))
        nchanges.append(n)


    # plot space correlations
    fig11 = plt.figure()
    n, bins, patches = plt.hist(space_corr, 100, range=[0,1450], facecolor='blue', alpha=0.5, align='mid')
    sns.despine()
    plt.xlabel('space correlation radius of pixel changes per user')
    plt.ylabel('number of users')
    plt.xlim([0,1450])
    plt.ylim([5, 1e6])
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'Space_correlation_radius_of_pixel_changes.png'), bbox_inches='tight')
    plt.close('all')

    # plot space correlations vs number of pixel changes per user
    fig12 = plt.figure()
    plt.hist2d(space_corr, nchanges, bins=(100,798), range=[[0,1450],[2,800]], cmap=plt.cm.jet, norm=colors.LogNorm(vmin=0.9, vmax=3e5))
    plt.xlabel('space correlation radius of pixel changes per user')
    plt.ylabel('number of pixel changes per user')
    plt.colorbar(label='number of users')
    plt.xlim([0,1450])
    plt.ylim([0.9, 800])
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'Space_correlation_radius_vs_numberof_pixel_changes.png'), bbox_inches='tight')

    # compute space correlation with the combinatorial method
    print('compute space correlations (combinatorial method)')
    space_corr2 = []
    for i in range(0, len(singleuser_limits)-1): 
        n = singleuser_limits[i+1] - singleuser_limits[i]
        if n < 2:
            continue
        x = np.array(xcoor[singleuser_limits[i]:singleuser_limits[i+1]], dtype=np.int32)
        y = np.array(ycoor[singleuser_limits[i]:singleuser_limits[i+1]], dtype=np.int32)
        sum = 0
        for j in range(0, n-1):
            xtmp = x[j+1:n]
            ytmp = y[j+1:n]
            sum += np.sum( np.sqrt( (xtmp-x[j])**2 + (ytmp-y[j])**2 ) ) # distances of all (xtmp,ytmp) points to point #j

        space_corr2.append( sum / (n*(n-1)) )

    # plot space correlations
    fig13 = plt.figure()
    n, bins, patches = plt.hist(space_corr2, 100, range=[0,1450], facecolor='blue', alpha=0.5, align='mid')
    sns.despine()
    plt.xlabel('correlation of positions of pixel changes per user')
    plt.ylabel('number of users')
    plt.xlim([0,1450])
    plt.ylim([5, 1e6])
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'Space_correlation_radius_of_pixel_changes_combinatorialMeth.png'), bbox_inches='tight')

    # plot space correlations vs number of pixel changes per user
    fig14 = plt.figure()
    plt.hist2d(space_corr2, nchanges, bins=(100,798), range=[[0,1450],[2,800]], cmap=plt.cm.jet, norm=colors.LogNorm(vmin=0.9, vmax=3e5))
    plt.xlabel('combinatorial space correlation of pixel changes per user')
    plt.ylabel('number of pixel changes per user')
    plt.colorbar(label='number of users')
    plt.xlim([0,1450])
    plt.ylim([0.9, 800])
    plt.yscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH,'Space_correlation_radius_vs_numberof_pixel_changes_combinatorialMeth.png'), bbox_inches='tight')
