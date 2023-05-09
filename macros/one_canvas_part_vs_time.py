import numpy as np
import os, sys
import rplacem.canvas_part as cp
import rplacem.transitions as trans
import rplacem.canvas_part_statistics as stat
import rplacem.early_warning_signals as ews
import rplacem.compute_variables as comp
import matplotlib.colors as pltcolors
import rplacem.utilities as util
import rplacem.variables_rplace2022 as var
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

'''
icp = 5 # index of the studied composition. If negative, the CanvasPart is redefined

if icp < 0: # not using the stored compositions
    x1 = 0
    x2 = 1999
    y1 = 0
    y2 = 1999

    canpart = cp.CanvasPart(border_path=[[[x1, y1], [x1, y2], [x2, y2], [x2, y1]]],
                            #id='000021', 
                            pixel_changes_all=None) 

else: # getting the composition from the stored ones
    file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle') 
    with open(file_path, 'rb') as f:
        canvas_parts = pickle.load(f)
    canpart = canvas_parts[icp]
    print(canpart)

'''


# only quick check here
pixel_changes_all = util.get_all_pixel_changes()
atlas, num = util.load_atlas()

'''
canvas_comps = []
file_path = os.path.join(var.DATA_PATH, 'CanvasParts_test.pickle')
with open(file_path, 'rb') as f:
    canvas_comps = pickle.load(f)
'''

'''
n = 300
isrect = np.zeros(n)
npix = np.zeros(n)
bmpsize_over_npix = np.zeros(n)
bmpsize_over_npix_rectequiv = np.zeros(n)

for i in range(0,n):
    print(i)
    print(atlas[i]['id'])
    print(atlas[i]['links'])
    #canvas_comps.append( cp.CanvasPart(id=atlas[i]['id'], pixel_changes_all=pixel_changes_all, atlas=atlas) )
    isrect[i] = canvas_comps[i].is_rectangle
    npix[i] = len(canvas_comps[i].coords[0])
    cpstat = stat.CanvasPartStatistics(canvas_comps[i], verbose=True, compute_vars={'stability': 1, 'mean_stability': 1, 'entropy' : 1}, dont_keep_dir=True)
    bmpsize_over_npix[i] = cpstat.bmpsize / cpstat.area
    bmpsize_over_npix_rectequiv[i] = cpstat.bmpsize / cpstat.area_rectangle


#with open(file_path, 'wb') as handle:
#    pickle.dump(canvas_comps,
#                handle,
#                protocol=pickle.HIGHEST_PROTOCOL)


# scatter plot
fig = plt.figure()
plt.scatter(npix, bmpsize_over_npix,  s=3)
plt.ylabel('bmp file size / # active pixels')
plt.xlabel('# active pixels')
plt.xscale('log')
plt.xlim(1,max(npix)*1.3)
plt.savefig(os.path.join(var.FIGS_PATH, 'bmpfilesize_over_npix.png'), bbox_inches='tight')

# scatter plot
fig = plt.figure()
plt.scatter(npix, bmpsize_over_npix_rectequiv,  s=3)
plt.ylabel('bmp file size / # active pixels')
plt.xlabel('# active pixels')
plt.xscale('log')
plt.xlim(1,max(npix)*1.3)
plt.savefig(os.path.join(var.FIGS_PATH, 'bmpfilesize_over_npix__rectangle_equivalent.png'), bbox_inches='tight')

# scatter plot
fig2 = plt.figure()
plt.scatter(npix[np.where(isrect)[0]], bmpsize_over_npix[np.where(isrect)[0]], s=3)
plt.ylabel('bmp file size / # active pixels')
plt.xlabel('# active pixels')
plt.xscale('log')
plt.xlim(1,max(npix)*1.3)
plt.savefig(os.path.join(var.FIGS_PATH, 'bmpfilesize_over_npix_onlyrectangles.png'), bbox_inches='tight')




tlims = np.arange(var.TIME_ENLARGE2-1, var.TIME_WHITEONLY,100)
comp.save_part_over_time(canpart, tlims,
                        record_pixels=False,
                        delete_bmp=True,
                        delete_png=False,
                        show_plot=False,
                        print_progress=False,
                        remove_inactive=True
                        )
util.save_movie(os.path.join(var.FIGS_PATH, 'twoztm','VsTime'), fps=10)

attdef = comp.num_changes_and_users(canpart,tlims,canpart.white_image(2),False,False)
print(attdef[11], len(attdef[11]))
print(attdef[12], len(attdef[12]))



file_path = os.path.join(var.DATA_PATH, 'canvas_composition_statistics_all.pickle') 
with open(file_path, 'rb') as f:
    cpstats = pickle.load(f)

for cps in cpstats:
    if cps.id != '000006':
        continue
    else:
        cpstatrec = cps
        break

'''

canpart = cp.CanvasPart(
                        #border_path=[[[299, 318], [299,450], [507,450], [507, 318]]], #'rectangle_299.318_to_507.45'
                        id='000006',#'twoztm',#'twwgx2',#'000006' 
                        pixel_changes_all=pixel_changes_all,
                        verbose=True, save=True)

cpstat = stat.CanvasPartStatistics(canpart, n_tbins=750, 
                                    compute_vars={'stability': 2, 'entropy' : 2, 'transitions' : 2, 'attackdefense' : 2},
                                    sliding_window=14400,
                                    verbose=False, dont_keep_dir=False)


#varchange = ews.ratio_to_slidingmean(cpstat.diff_stable_pixels_vst, cpstat.t_interval, slidingrange=21600)

#ews.ews_2Dsignificance_1comp(cpstat, cpstat.diff_stable_pixels_vst, 'differing_stable_pixels')
#ews.ews_2Dsignificance_allcomp(cpstats, warning_cooldown = 14400, ews_slidingwindow=4000)
#ews.ews_2Dsignificance_allcomp([cpstat], warning_cooldown = 14400, ews_slidingwindow=4000, singlecompsave=True)

#print(ews.firing_times(cpstat, cpstat.diff_stable_pixels_vst, 300, 27))

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.frac_pixdiff_inst_vs_stable_norm)
sns.despine()
plt.ylabel('fraction of pixels different from previous-step stable image / 5 min')
plt.xlabel('Time [s]')
plt.ylim([0, 1.1*max(cpstat.frac_pixdiff_inst_vs_stable_norm)])
plt.xlim([0, var.TIME_TOTAL])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'Fraction_of_differing_pixels_vs_stable_normalized.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.frac_pixdiff_inst_vs_inst_norm)
sns.despine()
plt.ylabel('fraction of pixels different from previous-step image / 5 min')
plt.xlabel('Time [s]')
plt.ylim([0, 1.1*max(cpstat.frac_pixdiff_inst_vs_inst_norm)])
plt.xlim([0, var.TIME_TOTAL])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'Fraction_of_differing_pixels_normalized.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.frac_pixdiff_inst_vs_ref)
sns.despine()
plt.ylabel('fraction of pixels different from sliding-window reference image')
plt.xlabel('Time [s]')
plt.ylim([0, 1.1*max(cpstat.frac_pixdiff_inst_vs_ref)])
plt.xlim([0, var.TIME_TOTAL])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'Fraction_of_differing_pixels_vs_slidingwindowref.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.frac_pixdiff_inst_vs_stable_norm, label='inst. vs stable / 5 min')
plt.plot(cpstat.t_ranges, cpstat.frac_pixdiff_inst_vs_inst_norm, label='inst. vs inst. / 5 min')
plt.plot(cpstat.t_ranges, cpstat.frac_pixdiff_inst_vs_ref, label='inst. vs sliding-window reference')
plt.plot(cpstat.t_ranges, cpstat.frac_pixdiff_stable_vs_ref, label='stable. vs sliding-window reference')
plt.xlabel('Time [s]')
plt.ylabel('fraction of differing pixels')
plt.yscale('log')
plt.xlim([0, var.TIME_TOTAL])
plt.ylim([1e-4, 1.1*np.amax(np.hstack((cpstat.frac_pixdiff_inst_vs_ref, 
                                    cpstat.frac_pixdiff_stable_vs_ref, 
                                    cpstat.frac_pixdiff_inst_vs_inst_norm, 
                                    cpstat.frac_pixdiff_inst_vs_stable_norm)))])
plt.legend()
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'Fraction_of_differing_pixels_variousmethods.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.instability_norm)
sns.despine()
plt.ylabel('instability / 5 min')
plt.xlabel('Time [s]')
plt.ylim([0, 1.1*max(cpstat.instability_norm)])
plt.xlim([0, var.TIME_TOTAL])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'instability_normalized.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.n_users_norm)
sns.despine()
plt.ylabel('# users / area / 5 min')
plt.xlabel('Time [s]')
plt.ylim([0, 1.1*max(cpstat.n_users_norm)])
plt.xlim([0, var.TIME_TOTAL])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'Number_of_users_normalized.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.n_changes_norm)
sns.despine()
plt.ylabel('# pixel changes / area / 5 min')
plt.xlabel('Time [s]')
plt.ylim([0, 1.1*max(cpstat.n_changes_norm)])
plt.xlim([0, var.TIME_TOTAL])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'Number_of_pixel_changes_normalized.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.entropy)
sns.despine()
plt.ylabel('entropy (computable information density)')
plt.xlabel('Time [s]')
plt.ylim([0, 1.1*max(cpstat.entropy)])
plt.xlim([0, var.TIME_TOTAL])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'entropy.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.frac_attack_changes)
sns.despine()
plt.ylabel('# attack changes / # defense changes')
plt.xlabel('Time [s]')
plt.ylim([0,1])
plt.hlines(y = 0.5, xmin=0, xmax=var.TIME_TOTAL, colors = 'black', linestyle='dashed')
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'Fraction_attack_pixelchanges.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.frac_attackonly_users)
sns.despine()
plt.ylabel('fraction of users only attacking')
plt.xlabel('Time [s]')
plt.ylim([0,1])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'fraction_of_users_onlyattacking.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.frac_defenseonly_users)
sns.despine()
plt.ylabel('fraction of users only defending')
plt.xlabel('Time [s]')
plt.ylim([0,1])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'fraction_of_users_onlydefending.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.frac_bothattdef_users)
sns.despine()
plt.ylabel('fraction of users both attacking and defending')
plt.xlabel('Time [s]')
plt.ylim([0,1])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'fraction_of_users_bothattackingdefending.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.returntime_median_overln2)
sns.despine()
plt.ylabel('median time for pixels to recover from attack [s] / ln(2)')
plt.xlabel('Time [s]')
plt.ylim([0, 1.1*max(cpstat.returntime_median_overln2[1:-2])])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'median_pixel_recovery_time.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.returntime_mean)
sns.despine()
plt.ylabel('mean time for pixels to recover from attack [s] / ln(2)')
plt.xlabel('Time [s]')
plt.ylim([0, 1.1*max(cpstat.returntime_mean[1:-2])])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'mean_pixel_recovery_time.png'), bbox_inches='tight')

plt.figure()
plt.plot(cpstat.t_ranges, cpstat.cumul_attack_timefrac)
sns.despine()
plt.ylabel('Fraction of the time that all pixels spent in an attack color [s]')
plt.xlabel('Time [s]')
plt.ylim([0, 1.1*max(cpstat.cumul_attack_timefrac[1:])])
plt.xlim([0, var.TIME_TOTAL])
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'attack_time_fraction_allpixels.png'), bbox_inches='tight')

'''
fig, ax = plt.subplots()
nbinsret = 100
tmaxret = 10000
returnt_bins = np.arange(0,tmaxret+tmaxret/nbinsret, tmaxret/nbinsret)
matrix = np.zeros((cpstat.n_t_bins, nbinsret))
for i in range(0,cpstat.n_t_bins):
    rettime = np.array(cpstat.returntime_tbinned[0][i])
    matrix[i] = np.histogram(rettime, returnt_bins)[0] if len(rettime) > 0 else np.zeros(nbinsret)
plt.pcolormesh( cpstat.t_ranges, 
                returnt_bins[:-1]+tmaxret/nbinsret/2, 
                np.transpose(matrix), 
                cmap=plt.cm.jet, norm=pltcolors.LogNorm(vmin=0.95, vmax=700))
plt.xlabel('time [s]')
plt.ylabel('time to recover from fresh attack to ref image [s]')
plt.ticklabel_format(style='scientific')
plt.colorbar(label='# fresh attacks')
plt.savefig(os.path.join(var.FIGS_PATH, canpart.out_name(), 'returntime_vs_time_hist2d.png'))


plt.figure()
plt.plot(cpstat.t_ranges, cpstat.returntime_mean[0])
sns.despine()
plt.ylabel('mean time for pixels to recover from fresh attack [s] / ln(2)')
plt.xlabel('Time [s]')
plt.yscale('log')
plt.ylim([8,3000])
plt.xlim([0000, 300000])
plt.vlines(x = [cpstat.transition_times[0][0], cpstat.transition_times[0][1], cpstat.transition_times[0][2], cpstat.transition_times[0][3]], ymin=0, ymax=1600, colors = 'black', linestyle='dashed')
plt.savefig(os.path.join(var.FIGS_PATH, canpart.out_name(), 'mean_pixel_recovery_time.png'))
'''

'''
ntrans = 0
for i in range(0,20):
    canpart = canvas_comps[i]
    print(i, canpart.id)
    cpstat = stat.CanvasPartStatistics(canpart, n_tbins=100, 
                                    compute_vars={'stability': 1, 'mean_stability': 1, 'entropy' : 1, 'transitions' : 1, 'attackdefense' : 1},
                                    verbose=False, dont_keep_dir=False)
    print(cpstat.transition_times)
    if len(cpstat.transition_times)>0:
        ntrans += 1

print('frac of compos with a trans = ',ntrans/200)


plt.figure()
plt.plot(cpstat.t_ranges[1:] - cpstat.t_interval / 2, cpstat.instability_vst_norm, label='instability ')
plt.plot(cpstat100.t_ranges[1:] - cpstat100.t_interval / 2, cpstat100.instability_vst_norm, label='instability (100 time bins)')
plt.plot(cpstat200.t_ranges[1:] - cpstat200.t_interval / 2, cpstat200.instability_vst_norm, label='instability (200 time bins)')
plt.plot(cpstat1000.t_ranges[1:] - cpstat1000.t_interval / 2, cpstat1000.instability_vst_norm, label='instability (1000 time bins)')
sns.despine()
plt.legend(loc="upper left")
plt.ylabel('instability / 5 min')
plt.xlabel('Time [s]')
plt.ylim([0.0001, max(cpstat1000.instability_vst_norm)*0.2])
plt.xlim([120000, 250000])
plt.yscale('log')
plt.savefig(os.path.join(var.FIGS_PATH, canpart.out_name(), 'instability_various_time_intervals.png'), bbox_inches='tight')
'''


'''
# create directories if they don't exist yet
try:
    os.makedirs(os.path.join(var.FIGS_PATH, canpart.out_name()))
except OSError: 
    pass
try:
    os.makedirs(os.path.join(var.FIGS_PATH, canpart.out_name(), 'VsTime'))
except OSError: 
    pass

# time-dependent images + file sizes and compression
time_intervals = np.arange(0, var.TIME_TOTAL, 300)
file_size_bmp, file_size_png, t_inds_list = comp.save_part_over_time(canpart, time_intervals, delete_bmp=True, delete_png=False, show_plot=False)
util.save_movie(os.path.join(var.FIGS_PATH, canpart.out_name(),'VsTime'), 15)
comp.plot_compression(file_size_bmp, file_size_png, time_intervals, out_name = canpart.out_name())

# mean stability 
mean_stability = comp.stability(canpart,np.asarray([0, var.TIME_WHITEONLY]), True,True,False,True)[0][0]
print('mean stability = ', mean_stability)

# time-dependent stability
time_bins = 80
time_interval = var.TIME_WHITEONLY / time_bins  # seconds
time_ranges = np.arange(0, var.TIME_WHITEONLY+time_interval-1e-4, time_interval)
stability_vs_time = comp.stability(canpart, time_ranges, False, False, 
                                   False, #icp < 0, 
                                   True)[0]

plt.figure()
plt.plot(time_ranges[:-1]+time_interval/2, stability_vs_time)
sns.despine()
plt.ylabel('stability')
plt.xlabel('Time [s]')
plt.ylim([0, 1])
plt.xlim([0, var.TIME_TOTAL])

plt.savefig(os.path.join(var.FIGS_PATH, canpart.out_name(), 'stability.png'))

transitions = comp.find_transitions(time_ranges, stability_vs_time)
print(transitions)
print(canpart.border_path_times)

'''
