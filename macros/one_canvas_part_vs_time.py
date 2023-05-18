import numpy as np
import os, sys
import rplacem.canvas_part as cp
import rplacem.transitions as trans
import rplacem.canvas_part_statistics as stat
import rplacem.early_warning_signals as ews
import rplacem.compute_variables as comp
import matplotlib.colors as pltcolors
import rplacem.utilities as util
import rplacem.plot_utilities as plot
import rplacem.variables_rplace2022 as var
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

pixel_changes_all = util.get_all_pixel_changes()
atlas, num = util.load_atlas()

# intro
fromatlas = True
cp_fromfile = False
cps_fromfile = False

id = '000297' #'twoztm',#'twwgx2',#'000006' # only if fromatlas
x1 = 0 # only if not fromatlas
x2 = 1999
y1 = 0
y2 = 1999

if not fromatlas:
    cp_fromfile = False
    cps_fromfile = False

# Get CanvasPart
if cp_fromfile:
    file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle') 
    with open(file_path, 'rb') as f:
        canvas_parts = pickle.load(f)
      
    for canp in canvas_parts:
        if canp.id != id:
            continue
        else:
            canpart = canp
            break

else:
    if not fromatlas:
        canpart = cp.CanvasPart(border_path=[[[x1, y1], [x1, y2], [x2, y2], [x2, y1]]],
                                pixel_changes_all=pixel_changes_all,
                                verbose=False, save=True)
    else:
        canpart = cp.CanvasPart(id=id,
                                pixel_changes_all=pixel_changes_all,
                                verbose=False, save=True)

# Get CanvasPartStatistics
if cps_fromfile:
    file_path = os.path.join(var.DATA_PATH, 'canvas_composition_statistics_all.pickle') 
    with open(file_path, 'rb') as f:
        cpstats = pickle.load(f)

    for cps in cpstats:
        if cps.id != id:
            continue
        else:
            cpstat = cps
            break

else: 
    cpstat = stat.CanvasPartStatistics(canpart, t_interval=300, 
                                        compute_vars={'stability': 3, 'entropy' :3, 'transitions' : 3, 'attackdefense' : 3, 'other' : 1},
                                        sliding_window=14400,
                                        verbose=False, dont_keep_dir=False)


cpstat.fill_timeseries_info()

cpstat.frac_pixdiff_inst_vs_stable_norm.plot1d(ymin=0)
cpstat.frac_pixdiff_inst_vs_inst_norm.plot1d(ymin=0)
cpstat.frac_pixdiff_inst_vs_swref.plot1d(ymin=0)
cpstat.frac_pixdiff_inst_vs_swref_forwardlook.plot1d(ymin=0)

plt.figure()
plt.plot(cpstat.t_lims, cpstat.frac_pixdiff_inst_vs_stable_norm.val, label='inst. vs stable / 5 min')
#plt.plot(cpstat.t_lims, cpstat.frac_pixdiff_inst_vs_inst_norm.val, label='inst. vs inst. / 5 min')
plt.plot(cpstat.t_lims, cpstat.frac_pixdiff_inst_vs_swref.val, label='inst. vs sliding-window ref')
plt.plot(cpstat.t_lims, cpstat.frac_pixdiff_inst_vs_swref_forwardlook.val, label='inst. vs fwd-looking sliding-window ref')
#plt.plot(cpstat.t_lims, cpstat.frac_pixdiff_stable_vs_swref.val, label='stable. vs sliding-window ref')
plt.xlabel('Time [s]')
plt.ylabel('fraction of differing pixels')
#plt.yscale('log')
plt.xlim([cpstat.frac_pixdiff_inst_vs_swref.tmin, var.TIME_TOTAL])
plt.ylim([1e-4, 1.1*np.amax(np.hstack((cpstat.frac_pixdiff_inst_vs_swref.val, 
                                    cpstat.frac_pixdiff_stable_vs_swref.val, 
                                    cpstat.frac_pixdiff_inst_vs_inst_norm.val, 
                                    cpstat.frac_pixdiff_inst_vs_stable_norm.val)))])
plt.legend(loc='upper right')
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'Fraction_of_differing_pixels_variousmethods.png'), dpi=200, bbox_inches='tight')

#plt.figure()
#plt.plot(cpstat.t_lims, cpstat.frac_pixdiff_inst_vs_stable_norm.val / cpstat.frac_pixdiff_inst_vs_inst_norm.val)
#plt.xlabel('Time [s]')
#plt.ylabel('inst. vs stable / inst vs inst')
#plt.xlim([0, var.TIME_TOTAL])
#plt.ylim([0., 2])
#plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'Fraction_of_differing_pixels__ratio_inst_vs_stable.png'), bbox_inches='tight')

cpstat.instability_norm.plot1d(ymin=0)
cpstat.n_users_norm.plot1d(ymin=0)
cpstat.n_changes_norm.plot1d(ymin=0)
cpstat.entropy.plot1d(ymin=0)
cpstat.frac_attack_changes.plot1d(ymin=0, ymax=1, hline=0.5)
cpstat.frac_attackonly_users.plot1d(ymin=0, ymax=1)
cpstat.frac_defenseonly_users.plot1d(ymin=0, ymax=1)
cpstat.frac_bothattdef_users.plot1d(ymin=0, ymax=1)
cpstat.returntime_median_overln2.plot1d(ymin=0, ibeg_remove=1, iend_remove=2)
cpstat.returntime_mean.plot1d(ymin=0, ibeg_remove=1, iend_remove=2)
cpstat.cumul_attack_timefrac.plot1d(ymin=0, ibeg_remove=1)

cpstat.frac_moderator_changes.plot1d(ymin=0)
cpstat.frac_cooldowncheat_changes.plot1d(ymin=0)
cpstat.frac_redundant_color_changes.plot1d(ymin=0)
cpstat.frac_redundant_coloranduser_changes.plot1d(ymin=0)

returnt_bins = np.arange(0, cpstat.sw_width_sec-1e-4, cpstat.returnt_binwidth)
plot.draw_2dplot(cpstat.t_lims, returnt_bins, cpstat.returntime_tbinned,
                 xlab='Time [s]', ylab='time to recover from fresh attack to ref image [s]', zlab='# fresh attacks',
                 ymax=cpstat.sw_width_sec*0.6,
                 logz=True, zmin=0.9, zmax=1000,
                 outname=os.path.join(cpstat.id, 'returntime_vs_time_hist2d.png'))

plot.cpstat_tseries(cpstat, nrows=7, ncols=2, figsize=(8,11.5), fontsize=10, save=True)

#OLD EWS
'''
varchange = ews.ratio_to_slidingmean(cpstat.diff_stable_pixels_vst, cpstat.t_interval, slidingrange=21600)

ews.ews_2Dsignificance_1comp(cpstat, cpstat.diff_stable_pixels_vst, 'differing_stable_pixels')
ews.ews_2Dsignificance_allcomp(cpstats, warning_cooldown = 14400, ews_slidingwindow=4000)
ews.ews_2Dsignificance_allcomp([cpstat], warning_cooldown = 14400, ews_slidingwindow=4000, singlecompsave=True)
print(ews.firing_times(cpstat, cpstat.diff_stable_pixels_vst, 300, 27))
'''

# STUDY OF BMP FILE SIZE

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
'''
