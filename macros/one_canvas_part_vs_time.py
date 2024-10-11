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
from rplacem import var as var
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# intro
fromatlas = True
cp_fromfile = True
cp_fromatlasfile = True
cps_fromfile = True

if not cp_fromfile:
    pixel_changes_all = util.get_all_pixel_changes()
    atlas, num = util.load_atlas()

id = 'txibq9'#'txkd33'#'u38eza' #'000297' #'twoztm',#'twwgx2',#'twpx5e' # only if fromatlas 

x1 = var.CANVAS_MINMAX[-1, 0, 0]
x2 = var.CANVAS_MINMAX[-1, 0, 1]
y1 = var.CANVAS_MINMAX[-1, 1, 0]
y2 = var.CANVAS_MINMAX[-1, 1, 1]


# Get CanvasPart
if not cps_fromfile:
    if cp_fromfile:
        if cp_fromatlasfile:
            file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle') 
            with open(file_path, 'rb') as f:
                canvas_parts = pickle.load(f)
            
            for canp in canvas_parts:
                if canp.info.id != str(id):
                    continue
                else:
                    canpart = canp
                    break
        else:
            file_path = os.path.join(var.DATA_PATH, 'CanvasPart_rectangle_'+str(x1)+'.'+str(y1)+'_to_'+str(x2)+'.'+str(y2)+'.pickle') 
            with open(file_path, 'rb') as f:
                canpart = pickle.load(f)
                f.close()

    else:
        if fromatlas:
            atlas_info_separated = cp.get_atlas_border(id=id, atlas=atlas, addtime_before=10*3600, addtime_after=4*3600)
            canvas_comps = []
            for ainfo in atlas_info_separated:
                # actual canvas composition here
                #print(ainfo.id, ainfo.border_path, ainfo.border_path_times)
                canvas_comps.append( cp.CanvasPart(atlas_info=ainfo, pixel_changes_all=pixel_changes_all, verbose=False, save=True) )
            canpart = canvas_comps[0]
        else:
            # here, assume the whole canvas is wanted
            info = cp.AtlasInfo(border_path=[[[x1, y1], [x1, y2], [x2, y2], [x2, y1]]])
            canpart = cp.CanvasPart(atlas_info=info,
                                    pixel_changes_all=pixel_changes_all,
                                    verbose=True, save=True)


# Get CanvasPartStatistics
if cps_fromfile:
    file_path = os.path.join(var.DATA_PATH, 'cpart_stats_sw3_ta0.35_tr6_2022.pkl') 
    with open(file_path, 'rb') as f:
        cpstats = pickle.load(f)
    
    for i,cps in enumerate(cpstats):
        if cps.id != id:
            continue
        else:
            cpstat = cps
            break
    
else: 
    cpstat = stat.CanvasPartStatistics(canpart, t_interval=300, #tmax=30000,
                                        compute_vars={'stability': 1, 'entropy' :3, 'transitions' : 1, 'attackdefense' : 1, 'other' : 1, 
                                                      'ews' : 1, 'inout':0, 'lifetime_vars':0, 'void_attack':0},
                                        sliding_window=int(3*3600), 
                                        verbose=True, dont_keep_dir=False, compression='DEFLATE_BMP_PNG', flattening='ravel')

savecpstat = False
if savecpstat:
    file_path = os.path.join(var.DATA_PATH, 'CanvasPartStatistics_'+ canpart.out_name() + '.pickle')
    with open(file_path, 'wb') as handle:
        pickle.dump(cpstat,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

print(cpstat.sw_width_sec)
print(np.any(cpstat.autocorr_bycase.val < 0))
cpstat.entropy.set_ratio_to_sw_average()
plt.figure()
plt.plot(cpstat.t_lims, cpstat.entropy.val)
plt.xlim([0,295000])
#plt.ylim([0,1])
plt.savefig(os.path.join(var.FIGS_PATH, str(cpstat.id), 'entropy'))

plt.figure()
plt.plot(cpstat.t_lims, cpstat.entropy.ratio_to_sw_mean)
plt.xlim([0,295000])
plt.ylim([0,3])
plt.savefig(os.path.join(var.FIGS_PATH, str(cpstat.id), 'entropy_ratio_to_sw_mean'))

plt.figure()
plt.plot(cpstat.t_lims, cpstat.frac_redundant_color_changes.val)
plt.xlim([0,295000])
plt.ylim([0,1])
plt.savefig(os.path.join(var.FIGS_PATH, str(cpstat.id), 'frac_redundant_color_changes'))

plt.figure()
plt.plot(cpstat.t_lims[cpstat.t_lims < 60000], cpstat.frac_pixdiff_inst_vs_swref.val[cpstat.t_lims < 60000])
plt.xlim([0,60000])
plt.ylim([0,1])
plt.savefig(os.path.join(var.FIGS_PATH, str(cpstat.id), 'frac_pixdiff_vs_swref'))

plt.figure()
plt.plot(cpstat.frac_pixdiff_inst_vs_inst_norm.val[1:])
plt.plot(np.abs(np.diff(cpstat.frac_pixdiff_inst_vs_swref.val)))
plt.ylim([0,0.05])
plt.savefig(os.path.join(var.FIGS_PATH, str(cpstat.id), 'frac_pixdiff_comparison'))

plt.figure()
plt.scatter(cpstat.frac_pixdiff_inst_vs_inst_norm.val[1:], np.abs(np.diff(cpstat.frac_pixdiff_inst_vs_swref.val)))
plt.xlim([0,0.02])
plt.ylim([0,0.02])
plt.savefig(os.path.join(var.FIGS_PATH, str(cpstat.id), 'frac_pixdiff_comparison_scatter'))

def printrho(v1, v2, v1n, v2n):
    print('rho for ',v1n,' and ',v2n,' = ', np.corrcoef(v1[v1<1.1*np.max(v1)], v2[v1<1.1*np.max(v1)])[0,1])
    plt.figure()
    plt.scatter(v1,v2)
    plt.ylabel(v2n)
    plt.xlabel(v1n)
    plt.savefig(os.path.join(var.FIGS_PATH, str(cpstat.id), 'test_correlation_'+v1n+'_'+v2n))
    plt.close()

print('rhos')

printrho(cpstat.variance2.val, cpstat.variance_multinom.val, 'variance2', 'variance_multinom')
printrho(cpstat.variance2.val, cpstat.variance_subdom.val, 'variance2', 'variance_subdom')
printrho(cpstat.variance2.val, cpstat.instability_norm[0].val, 'variance2', 'instability_norm')
printrho(cpstat.variance_multinom.val, cpstat.variance_subdom.val,'variance_multinom','variance_subdom')
printrho(cpstat.variance2.val, cpstat.autocorr_multinom.val, 'variance2', 'autocorr_multinom')
printrho(cpstat.autocorr_bycase.val, cpstat.autocorr_bycase_norm.val, 'autocorr_bycase', 'autocorr_bycase_norm')
printrho(cpstat.autocorr_bycase.val, cpstat.autocorr_dissimil.val, 'autocorr_bycase', 'autocorr_dissimil')
printrho(cpstat.autocorr_bycase.val, cpstat.autocorr_subdom.val, 'autocorr_bycase', 'autocorr_subdom')
printrho(cpstat.autocorr_bycase.val, cpstat.autocorr_multinom.val, 'autocorr_bycase', 'autocorr_multinom')
printrho(cpstat.autocorr_multinom.val, cpstat.autocorr_subdom.val, 'autocorr_multinom', 'autocorr_subdom')
printrho(cpstat.autocorr_multinom.val, cpstat.autocorr_dissimil.val, 'autocorr_multinom', 'autocorr_dissimil')

print('figs')

cpstat.fill_timeseries_info()
for cpstat in [cpstat]:#cpstats
    if cpstat.n_transitions == 0:
        continue
    print(cpstat.id)

    try:
        os.mkdir(os.path.join(var.FIGS_PATH, str(cpstat.id)))
    except:
        ''

    cpstat.frac_pixdiff_inst_vs_stable_norm.plot1d(ymin=0)
    cpstat.frac_pixdiff_inst_vs_inst_norm.plot1d(ymin=0)
    cpstat.frac_pixdiff_inst_vs_swref.plot1d(ymin=0)
    cpstat.frac_pixdiff_inst_vs_swref_forwardlook.plot1d(ymin=0)

    for j in range(cpstat.n_transitions):
        plt.figure()
        plt.plot(cpstat.t_lims, cpstat.frac_pixdiff_inst_vs_swref.val)
        plt.plot(cpstat.t_lims, cpstat.frac_pixdiff_inst_vs_swref.val / cpstat.frac_pixdiff_inst_vs_swref.ratio_to_sw_mean)
        plt.xlim([cpstat.transition_times[j][0] - 10000, cpstat.transition_times[j][1] + 15000])
        plt.ylim([0,1])
        trans_start_t = trans.transition_start_time_simple(cpstat)[j]

        plt.vlines([trans_start_t, cpstat.transition_times[j][0], cpstat.transition_times[j][1]], 0, 1, colors=['black'], linestyles=['dotted'])
        plt.savefig(os.path.join(var.FIGS_PATH, str(cpstat.id),  'frac_pixdiff_inst_vs_swref_transition'+str(j)))
        plt.close()

        plt.figure()
        plt.plot(cpstat.t_lims, cpstat.frac_pixdiff_inst_vs_swref.ratio_to_sw_mean)
        plt.xlim([cpstat.transition_times[j][0] - 10000, cpstat.transition_times[j][1] + 15000])
        trans_start_t = trans.transition_start_time_simple(cpstat)[j]
        plt.vlines([trans_start_t, cpstat.transition_times[j][0], cpstat.transition_times[j][1]], 0, 10, colors=['black'], linestyles=['dotted'])
        plt.savefig(os.path.join(var.FIGS_PATH, str(cpstat.id),  'frac_pixdiff_inst_vs_swref_ratio_to_sw_mean_transition'+str(j)))
        plt.close()

        plt.figure()
        plt.plot(cpstat.t_lims, cpstat.frac_pixdiff_inst_vs_swref_forwardlook.val)
        plt.xlim([cpstat.transition_times[j][0] - 10000, cpstat.transition_times[j][1] + 15000])
        plt.ylim([0,1])
        trans_start_t = trans.transition_start_time_simple(cpstat)[j]
        plt.vlines([trans_start_t, cpstat.transition_times[j][0], cpstat.transition_times[j][1]], 0, 1, colors=['black'], linestyles=['dotted'])
        plt.savefig(os.path.join(var.FIGS_PATH, str(cpstat.id),  'frac_pixdiff_inst_vs_swref_forwardlook_transition'+str(j)))
        plt.close()

print('other figs')

itmin = np.argmax(cpstat.t_lims >= cpstat.tmin)
plt.figure()
plt.plot(cpstat.t_lims[itmin:], cpstat.frac_pixdiff_inst_vs_stable_norm.val[itmin:], label='inst. vs stable / 5 min')
#plt.plot(cpstat.t_lims[itmin:], cpstat.frac_pixdiff_inst_vs_inst_norm.val[itmin:], label='inst. vs inst. / 5 min')
plt.plot(cpstat.t_lims[itmin:], cpstat.frac_pixdiff_inst_vs_swref.val[itmin:], label='inst. vs sliding-window ref')
plt.plot(cpstat.t_lims[itmin:], cpstat.frac_pixdiff_inst_vs_swref_forwardlook.val[itmin:], label='inst. vs fwd-looking sliding-window ref')
#plt.plot(cpstat.t_lims[itmin:], cpstat.frac_pixdiff_stable_vs_swref.val[itmin:], label='stable. vs sliding-window ref')
plt.xlabel('Time [s]')
plt.ylabel('fraction of differing pixels')
#plt.yscale('log')
plt.xlim([cpstat.frac_pixdiff_inst_vs_swref.tmin, var.TIME_TOTAL])
plt.ylim([1e-4, 1.1*np.amax(np.hstack((cpstat.frac_pixdiff_inst_vs_swref.val[itmin:], 
                                    cpstat.frac_pixdiff_stable_vs_swref.val[itmin:], 
                                    cpstat.frac_pixdiff_inst_vs_inst_norm.val[itmin:], 
                                    cpstat.frac_pixdiff_inst_vs_stable_norm.val[itmin:])))])
plt.legend(loc='upper right')
plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'Fraction_of_differing_pixels_variousmethods.png'), dpi=200, bbox_inches='tight')

#plt.figure()
#plt.plot(cpstat.t_lims, cpstat.frac_pixdiff_inst_vs_stable_norm.val / cpstat.frac_pixdiff_inst_vs_inst_norm.val)
#plt.xlabel('Time [s]')
#plt.ylabel('inst. vs stable / inst vs inst')
#plt.xlim([0, var.TIME_TOTAL])
#plt.ylim([0., 2])
#plt.savefig(os.path.join(var.FIGS_PATH, cpstat.id, 'Fraction_of_differing_pixels__ratio_inst_vs_stable.png'), bbox_inches='tight')

cpstat.instability_norm[0].plot1d(ymin=0)
cpstat.n_users_norm.plot1d(ymin=0)
cpstat.n_changes_norm.plot1d(ymin=0)
#cpstat.entropy.plot1d(ymin=0)
cpstat.frac_attack_changes.plot1d(ymin=0, ymax=1, hline=0.5)
cpstat.frac_attackonly_users.plot1d(ymin=0, ymax=1)
cpstat.frac_defenseonly_users.plot1d(ymin=0, ymax=1)
cpstat.frac_bothattdef_users.plot1d(ymin=0, ymax=1)
cpstat.returntime[1].plot1d(ymin=0, ibeg_remove=1, iend_remove=2)
cpstat.returntime[0].plot1d(ymin=0, ibeg_remove=1, iend_remove=2)
cpstat.cumul_attack_timefrac.plot1d(ymin=0, ibeg_remove=1)

cpstat.frac_moderator_changes.plot1d(ymin=0)
cpstat.frac_cooldowncheat_changes.plot1d(ymin=0)
cpstat.frac_redundant_color_changes.plot1d(ymin=0)
cpstat.frac_redundant_coloranduser_changes.plot1d(ymin=0)

plot.cpstat_tseries(cpstat, nrows=11, ncols=2, figsize=(8,11.5), fontsize=10, save=True)

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
