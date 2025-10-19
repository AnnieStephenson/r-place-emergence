import os
import sys
import numpy as np
from rplacem import var
sys.path.append('/home/guillaumefa/r-place-emergence/rplacem/machine_learning')
import pickle
import EvalML
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, rankdata

vars_tocut = ['frac_pixdiff_inst_vs_swref_t-0-0',
            'frac_pixdiff_inst_vs_swref_t-1-1',
            'frac_pixdiff_inst_vs_swref_t-3-2',
            'frac_pixdiff_inst_vs_swref_t-5-4',
            'frac_pixdiff_inst_vs_swref_t-8-6',
            'frac_pixdiff_inst_vs_swref_t-12-9',
            'frac_pixdiff_inst_vs_swref_t-17-13',
            'frac_pixdiff_inst_vs_swref_t-24-18',
            'frac_pixdiff_inst_vs_swref_t-39-25'
            ]
weights_tocut = np.array([1,1,2,2,3,4,5,7,11])

def reject_messy_times(inputvals, varnames):
    threshold = 0.35/6 #0.06
    vars_idx = []
    for vtest in vars_tocut:
        for v in range(len(varnames)):
            if varnames[v] == vtest:
                vars_idx.append(v)
    print(inputvals.shape[0], 'instances entering reject_messy_times')
    
    vals_stabvars = inputvals[:, vars_idx]
    # for each instance, multiply weights and values of features element by element, then sum over features
    frac_pixdiff_sw = np.sum(np.multiply(weights_tocut, vals_stabvars), axis=1) / np.sum(weights_tocut)
    inds = np.where(frac_pixdiff_sw < threshold)[0]
    print(len(inds), 'instances exiting reject_messy_times')
    
    return inds


param_num = 0 # 0 is nominal result, 1 to 6 are sensitivity analysis #0,1,2,6
param_str = var.param_str_fun(param_num)

file_path = os.path.join(var.DATA_PATH, 'training_data_584variables_'+param_str+'.pickle') #3h-SW_widertimefromstart.pickle
with open(file_path, 'rb') as f:
    [inputvals, outputval, varnames, eventtime, id_idx, id_dict,
     coarse_timerange, 
     kendall_tau,
     n_traintimes, n_traintimes_coarse,
     safetimemargin
     ] = pickle.load(f) 

stable_inds = reject_messy_times(inputvals, varnames)
inputvals = inputvals[stable_inds]
outputval = outputval[stable_inds]
eventtime = eventtime[stable_inds]
id_idx = id_idx[stable_inds]

#print(varnames)
n = inputvals.shape[0]

ml_param = EvalML.AlgoParam(type='regression', # or 'classification
                          test2023=False,
                          n_features=None,
                          num_rounds=None, 
                          learning_rate=0.035, 
                          max_depth=8, 
                          min_child_weight=None, 
                          subsample=0.8, #0.8
                          colsample=0.75, #0.75
                          log_subtract_transform=1.5,
                          weight_highEarliness=0.4,
                          calibrate_pred=True)

earliness = -ml_param.transform_target(-outputval)

def plot_1D(featx, xmin=None):
    x = earliness if featx[0]=='earliness' else inputvals[:, np.where(varnames == featx[0])[0][0]]
    plt.figure()
    plt.hist(x, bins=100)
    plt.xlabel(featx[0] if featx[1] is None else featx[1])
    plt.ylabel('count')
    nameout = featx[0] if featx[2] is None else featx[2]
    if xmin is not None:
        plt.xlim(xmin, None)
    plt.savefig(os.path.join(var.FIGS_PATH, 'trainingvars_scatter', nameout+'_variation'+str(param_num)+'.png'))

def plot_xy(featx, featy,
            nbins=70,
            scat=False,
            meanx=False, #mean along x (of y values)
            meany=False, #mean along y (of x values)
            percentilex=False,
            percentiley=False,
            xmax=None, ymax=None, xmin=None, ymin=None,
            sel=None, mult=False, oneminus=False, inverty=False,
            ):

    diffx, diffy = False, False
    if type(featx[0]) == tuple:
        diffx = True
        x1 = earliness if featx[0][0]=='earliness' else inputvals[:, np.where(varnames == featx[0][0])[0][0]]
        x2 = earliness if featx[0][1]=='earliness' else inputvals[:, np.where(varnames == featx[0][1])[0][0]]
        x = np.multiply(1-x1 if oneminus else x1, x2) if mult else x1 - x2
    else:
        x = earliness if featx[0]=='earliness' else inputvals[:, np.where(varnames == featx[0])[0][0]]
    if type(featy[0]) == tuple:
        diffy = True
        y1 = earliness if featy[0][0]=='earliness' else inputvals[:, np.where(varnames == featy[0][0])[0][0]]
        y2 = earliness if featy[0][1]=='earliness' else inputvals[:, np.where(varnames == featy[0][1])[0][0]]
        y = np.multiply(1-y1 if oneminus else y1, y2) if mult else y1 - y2
    else:
        y = earliness if featy[0]=='earliness' else inputvals[:, np.where(varnames == featy[0])[0][0]]
        if inverty:
            y = 1/np.where(y<4e-2, 4e-2, y)

    if sel is not None:
        x = x[sel]
        y = y[sel]

    if featy[0] == "log(area)":
        x = x / np.power(10, 0.5*y) # divide by sqrt(area) to get a more linear relation

    xlab = (featx[0] if diffx else featx[0][0]) if featx[1] is None else featx[1]
    ylab = (featy[0] if diffy else featy[0][0]) if featy[1] is None else featy[1]

    # display as percentile of the variable rather than the variable itself
    if percentilex:
        x = rankdata(x) / n
        xlab += ' (percentile)'
    if percentiley:
        y = rankdata(y) / n
        ylab += ' (percentile)'

    x[np.abs(x) > 1e10] = 0
    y[np.abs(y) > 1e10] = 0
    if xmax is not None:
        x[x>xmax] = xmax
    if ymax is not None:
        y[y>ymax] = ymax

    plt.figure()
    if scat:
        plt.scatter(x, y, s=0.1, alpha=0.2)
    else:
        _, xedges, yedges, _ = plt.hist2d(x, y, bins=nbins, norm=mpl.colors.LogNorm())
        plt.colorbar()

    meanxval, meanyval = None, None

    if meanx:
        meanxval, _, _ = binned_statistic(x, y, statistic='mean', bins=xedges)
        meanxval = np.nan_to_num(meanxval, nan=(0 if xmin is None else xmin))
        bin_centers = xedges[1:] - (xedges[1] - xedges[0])/2
        if featy[0]=='earliness':
            meanxval = np.min(meanxval) + 10*(meanxval - np.min(meanxval))
        plt.plot(bin_centers, meanxval*(10 if featy[0]=='earliness' else 1), 'r-',
                 lw=2, label='mean'+(' x10' if featy[0]=='earliness' else ''))
    if meany:
        meanyval, _, _ = binned_statistic(y, x, statistic='mean', bins=yedges)
        meanyval = np.nan_to_num(meanyval, nan=(0 if ymin is None else ymin))
        bin_centers = yedges[1:] - (yedges[1] - yedges[0])/2
        if featx[0]=='earliness':
            meanyval = np.min(meanyval) + 10*(meanyval - np.min(meanyval))
        plt.plot(meanyval, bin_centers, 'r-', 
                 lw=2, label='mean'+(' x10' if featx[0]=='earliness' else ''))

    if meanx or meany:
        plt.legend()
    namex = featx[0] if featx[2] is None else featx[2]
    namey = featy[0] if featy[2] is None else featy[2]
    nameout = namex + ('_percentile' if percentilex else '') +'_vs_'+namey + ('_percentile' if percentiley else '')+('_scatter' if scat else '')

    if xmax is not None:
        plt.xlim(0 if xmin is None else xmin, xmax)
    if ymax is not None:
        plt.ylim(0 if ymin is None else ymin, ymax)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(os.path.join(var.FIGS_PATH, 'trainingvars_scatter', nameout+'_variation'+str(param_num)+'.png'))

    print(featx[2], featy[2], np.corrcoef(x, y)[0,1])

    return x, y, meanxval, meanyval

plot_1D(('image_shift_minpos_t-0-0', 'image_shift_minpos', 'image_shift_minpos'))
plot_1D(('image_shift_min_t-0-0', 'image_shift_min', 'image_shift_min'),-1)
plot_1D(('returntime_mean_t-0-0', 'return time mean 0-0', 'returntime_mean'))
plot_1D(('returntime_mean_t-84-70', 'return time mean 84-70', 'returntime_mean_84-70'))

plot_xy(('earliness','-log(time to transition)','earliness'), 
        (('frac_pixdiff_inst_vs_stable_t-0-0', 'frac_pixdiff_inst_vs_stable_t-39-25'), 'fraction differing pixels inst (0-0) - (39-25)', 'fracpixdiffinst_0-0--39-25'),
        nbins=80, meany=True, ymin=-0.3, ymax=0.3)

#plot_xy(('dist_average_norm_t-0-0', 'norm. average distance between changes', 'dist_changes'), 
#        ('log(area)', 'log(area)', 'logarea'),
#        nbins=120, meanx=True, meany=False,xmin=0,xmax=3)

plot_xy(('ripley_norm_d=2_t-0-0', 'ripley d=2', 'ripley_d=2'),
        ('log(area)', 'log(area)', 'logarea'),
        nbins=120, meanx=True, meany=False,xmin=0,xmax=4)

plot_xy(('image_shift_minpos_t-0-0', 'image_shift_minpos', 'image_shift_minpos'),
        ('log(area)', 'log(area)', 'logarea'),
        nbins=120, meanx=True, meany=False,xmin=0)

plot_xy(('frac_pixdiff_inst_vs_inst_downscaled2_t-0-0', 'frac_pixdiff_inst_vs_inst_downscaled2', 'frac_pixdiff_inst_vs_inst_downscaled2'),
        ('frac_pixdiff_inst_vs_inst_downscaled4_t-0-0', 'frac_pixdiff_inst_vs_inst_downscaled4', 'frac_pixdiff_inst_vs_inst_downscaled4'),
        nbins=120, meanx=False, meany=False,xmin=0,xmax=0.35,ymin=0,ymax=0.35)

plot_xy(('wavelet_high_to_low_t-0-0', 'wavelet_high_to_low', 'wavelet_high_to_low'),
        ('wavelet_mid_to_low_t-0-0', 'wavelet_mid_to_low', 'wavelet_mid_to_low'),
        nbins=120, meanx=False, meany=False,xmin=0)

plot_xy(('wavelet_high_to_mid_t-0-0', 'wavelet_high_to_mid', 'wavelet_high_to_mid'),
        ('wavelet_high_to_low_t-0-0', 'wavelet_high_to_low', 'wavelet_high_to_low'),
        nbins=120, meanx=False, meany=False,xmin=0)

plot_xy(('wavelet_high_to_mid_sw', 'wavelet_high_to_mid_sw', 'wavelet_high_to_mid_sw'),
        ('wavelet_high_to_low_sw', 'wavelet_high_to_low_sw', 'wavelet_high_to_low_sw'),
        nbins=120, meanx=False, meany=False,xmin=0,xmax=100,ymin=0,ymax=100)

plot_xy(('wavelet_high_to_mid_sw', 'wavelet_high_to_mid_sw', 'wavelet_high_to_mid_sw'),
        ('wavelet_mid_to_low_sw', 'wavelet_mid_to_low_sw', 'wavelet_mid_to_low_sw'),
        nbins=120, meanx=False, meany=False,xmin=0,xmax=100,ymin=0,ymax=100)
plot_xy(('wavelet_high_to_low_sw', 'wavelet_high_to_low_sw', 'wavelet_high_to_low_sw'),
        ('wavelet_mid_to_low_sw', 'wavelet_mid_to_low_sw', 'wavelet_mid_to_low_sw'),
        nbins=120, meanx=False, meany=False,xmin=0,xmax=100,ymin=0,ymax=100)

sys.exit()

plot_xy(('earliness','-log(time to transition)','earliness'), 
        (('frac_pixdiff_inst_vs_stable_t-0-0', 'frac_pixdiff_inst_vs_stable_t-84-70'), 'fraction differing pixels inst (0-0) - (84-70)', 'fracpixdiffinst_0-0--84-70'),
        nbins=80, meany=True, ymin=-0.3, ymax=0.3)

x,y,_,_ = plot_xy(('earliness','-log(time to transition)','earliness'), 
        (('frac_pixdiff_inst_vs_stable_t-0-0', 'frac_pixdiff_inst_vs_stable_t-24-18'), 'fraction differing pixels inst (0-0) - (24-18)', 'fracpixdiffinst_0-0--24-18'),
        nbins=80, meany=True, ymin=-0.3, ymax=0.3)

# check what happens for csu time instances
'''
inds_csu = np.where((y < -0.035) & (x > -2.7))[0]
compid_csu = id_dict[id_idx[inds_csu]]
print(len(inds_csu))
print(len(np.unique(compid_csu)))

print(compid_csu[:100])
print(eventtime[inds_csu][:100])
print(outputval[inds_csu][:100])

inds_stable_test = inds_csu[compid_csu == 'tx5hwp_part1']
print(inds_stable_test)
print(id_dict[id_idx[inds_stable_test]])

res = np.empty((len(vars_tocut), len(inds_stable_test)))
for iv,v in enumerate(vars_tocut):
    res[iv] = inputvals[inds_stable_test, np.where(varnames == v)[0][0]]
print(res.T)
print(np.sum(np.multiply(res.T[0], weights_tocut))/np.sum(weights_tocut))
'''

plot_xy(('returntime_mean_t-0-0', 'return time mean 0-0', 'returntime_mean'), 
        ('frac_pixdiff_inst_vs_swref_t-0-0', 'fraction differing pixels 0-0', 'fracpixdiff_0-0'),
        meanx=True,meany=True,
        ymax=0.35, xmax=4500,
        nbins=200)

retrate = inputvals[:, np.where(varnames == 'returnrate_t-0-0')[0][0]]
plot_xy(('returntime_mean_t-0-0', 'return time mean 0-0', 'returntime_mean'), 
        ('returnrate_t-0-0', '1 / (return rate 0-0)', 'invreturnrate_0-0'), 
        inverty=True, sel=((retrate>0)&(retrate<1)),
        meanx=True,meany=True,
        xmax=4500, ymax=27,
        nbins=100)
plot_xy(('returntime_mean_t-0-0', 'return time mean 0-0', 'returntime_mean'), 
        ('returnrate_t-0-0', 'return rate 0-0', 'returnrate_0-0'), 
        sel=((retrate>0)&(retrate<1)),
        meanx=True,meany=True,
        xmax=4500, ymax=1.01,
        nbins=100)

plot_xy(('returntime_mean_t-84-70', 'return time mean 84-70', 'returntime_mean_84-70'), 
        ('frac_pixdiff_inst_vs_swref_t-84-70', 'fraction differing pixels 84-70', 'fracpixdiff_84-70'),
        meanx=True,meany=True,
        ymax=0.65, xmax=4500,
        nbins=200)

plot_xy(('returntime_mean_t-0-0', 'return time mean 0-0', 'returntime_mean'), 
        ('frac_pixdiff_inst_vs_stable_t-0-0', 'fraction differing pixels inst 0-0', 'fracpixdiffinst_0-0'),
        meanx=True,meany=True,
        ymax=0.15, xmax=4500,
        nbins=200)

plot_xy(('autocorr_bycase_t-0-0', 'autocorrelation by case 0-0', 'autocorr_bycase'), 
        ('variance_frac_pixdiff_inst_t-0-0', 'variance 0-0', 'variance'),
        xmin=-0.02, xmax=0.02, ymax=0.011, meanx=True,
        nbins=200)

plot_xy(('autocorr_bycase_t-0-0', 'autocorrelation by case 0-0', 'autocorr_bycase'), 
        ('frac_pixdiff_inst_vs_swref_t-0-0', 'fraction differing pixels 0-0', 'fracpixdiff_0-0'),
        xmin=-0.02, xmax=0.02, ymax=0.3, meanx=True,
        nbins=200)

plot_xy(('autocorr_bycase_t-0-0', 'autocorrelation by case 0-0', 'autocorr_bycase'), 
        ('frac_pixdiff_inst_vs_stable_t-0-0', 'fraction differing pixels inst 0-0', 'fracpixdiffinst_0-0'),
        xmin=-0.02, xmax=0.02, ymax=0.3, meanx=True,
        nbins=200)

plot_xy(('autocorr_bycase_t-0-0', 'autocorrelation by case 0-0', 'autocorr_bycase'), 
        ('returnrate_t-0-0', 'return rate 0-0', 'returnrate_0-0'),
        xmin=-0.015, xmax=0.015, meanx=True,meany=True,
        nbins=300)

plot_xy(('earliness','-log(time to transition)','earliness'), 
        ('autocorr_bycase_t-0-0', 'autocorrelation by case 0-0', 'autocorr_bycase'), 
        meany=True, ymin=-0.03, ymax=0.03,
        nbins=300)

plot_xy(('earliness','-log(time to transition)','earliness'), 
        ('autocorr_subdom_t-0-0', 'autocorrelation sumdominant prod 0-0', 'autocorr_subdom'), 
        meany=True, ymax=0.003,
        nbins=300)

plot_xy(('autocorr_subdom_t-0-0', 'autocorrelation sumdominant prod 0-0', 'autocorr_subdom'), 
        ('frac_pixdiff_inst_vs_stable_t-0-0', 'fraction differing pixels inst 0-0', 'fracpixdiffinst_0-0'),
        meanx=True, ymax=0.3, xmax=0.003,
        nbins=200)

plot_xy(('returntime_mean_t-0-0', 'return time 0-0', 'returntime'), 
        ('frac_attack_changes_t-0-0', 'fraction attack changes 0-0', 'fracattack'),
        meanx=True, meany=True, xmax=4500,
        nbins=70)

plot_xy(('returntime_mean_t-84-70', 'return time 84-70', 'returntime_84-70'), 
        ('frac_attack_changes_t-84-70', 'fraction attack changes 84-70', 'fracattack_84-70'),
        meanx=True, meany=True, xmax=4500,
        nbins=70)

plot_xy(('returntime_mean_t-24-14', 'return time 24-14', 'returntime_24-14'), 
        ('frac_attack_changes_t-24-18', 'fraction attack changes 24-18', 'fracattack_24-18'),
        meanx=True, meany=True, xmax=4500,
        nbins=70)

plot_xy(('returntime_mean_t-0-0', 'return time 0-0', 'returntime'), 
        ('n_changes_norm_t-0-0', '# changes 0-0', 'nchanges'),
        meanx=True, meany=True,
        ymax=1.8, xmax=4500,
        nbins=200)

plot_xy(('returnrate_t-0-0', 'return rate 0-0', 'returnrate'), 
        ('n_changes_norm_t-0-0', '# changes 0-0', 'nchanges'),
        meanx=True, meany=True,
        ymax=1.8, 
        nbins=200)

plot_xy(('returntime_mean_t-84-70', 'return time 84-70', 'returntime_84-70'), 
        ('n_changes_norm_t-84-70', '# changes 84-70', 'nchanges_84-70'),
        meanx=True, meany=True,
        ymax=1.8, xmax=4500,
        nbins=200)

plot_xy(('returntime_mean_t-0-0', 'return time 0-0', 'returntime'), 
        (('frac_attack_changes_t-0-0', 'n_changes_norm_t-0-0'), 'number of attack changes', 'nattacks_0-0'),
        meanx=True, meany=True,
        xmax=4500, ymax=0.8, mult=True,
        nbins=200)


plot_xy(('returnrate_t-0-0', 'return rate 0-0', 'returnrate'), 
        ('frac_attack_changes_t-0-0', 'fraction attack changes 0-0', 'fracattack'),
        meanx=True, meany=True,
        nbins=200)

plot_xy(('returnrate_t-0-0', 'return rate 0-0', 'returnrate'), 
        ('frac_pixdiff_inst_vs_swref_t-0-0', 'fraction differing pixels 0-0', 'fracpixdiff_0-0'),
        meanx=True, meany=True,
        xmax=0.8, ymax=0.35, 
        nbins=200)

plot_xy(('returnrate_t-0-0', 'return rate 0-0', 'returnrate'), 
        ('frac_pixdiff_inst_vs_stable_t-0-0', 'fraction differing pixels inst 0-0', 'fracpixdiffinst_0-0'),
        meanx=True, meany=True,
        xmax=0.8, ymax=0.15, 
        nbins=200)

plot_xy(('returntime_mean_t-0-0', 'return time 0-0', 'returntime'), 
        (('frac_attack_changes_t-0-0', 'n_changes_norm_t-0-0'), 'number of defense changes', 'ndefenses_0-0'),
        meanx=True, meany=True,
        xmax=4500, ymax=0.8, 
        mult=True, oneminus=True,
        nbins=200)



plot_xy(('earliness','-log(time to transition)','earliness'), 
        ('frac_pixdiff_inst_vs_swref_t-0-0', 'fraction differing pixels 0-0', 'fracpixdiff_0-0'),
        nbins=80)

plot_xy(('earliness','-log(time to transition)','earliness'), 
        ('frac_pixdiff_inst_vs_swref_t-39-25', 'fraction differing pixels 39-25', 'fracpixdiff_39-25'),
        nbins=80,
        meany=True,
        percentiley=True,)

plot_xy(('earliness','-log(time to transition)','earliness'), 
        ('frac_pixdiff_inst_vs_swref_t-39-25', 'fraction differing pixels 39-25', 'fracpixdiff_39-25'),
        nbins=80,
        meany=True)

plot_xy(('frac_pixdiff_inst_vs_stable_t-69-55', 'fraction differing pixels inst 69-55', 'fracpixdiffinst_69-55'),
        ('n_users_sw_t-69-55', 'number of users 69-55', 'nusers_69-55'),
        nbins=80,
        meanx=True, xmax=0.4, ymax=1.3)

plot_xy(('frac_pixdiff_inst_vs_stable_t-84-70', 'fraction differing pixels inst 84-70', 'fracpixdiffinst_84-70'),
        ('n_users_sw_t-84-70', 'number of users 84-70', 'nusers_84-70'),
        nbins=80,
        meanx=True, xmax=0.4, ymax=1.3)

plot_xy(('frac_pixdiff_inst_vs_stable_t-84-70', 'fraction differing pixels 84-70', 'fracpixdiffinst_84-70'),
        ('n_users_sw_t-84-70', 'number of users 84-70', 'nusers_84-70_closetotransition'),
        nbins=80,
        meanx=True, xmax=0.4, ymax=1.3, sel=(outputval>-10000))

plot_xy(('frac_pixdiff_inst_vs_stable_t-54-40', 'fraction differing pixels inst 54-40', 'fracpixdiffinst_54-40'),
        ('n_users_sw_t-54-40', 'number of users 54-40', 'nusers_54-40'),
        nbins=80,
        meanx=True, xmax=0.4, ymax=1.3)

plot_xy(('frac_pixdiff_inst_vs_stable_t-54-40', 'fraction differing pixels inst 54-40', 'fracpixdiffinst_54-40'),
        ('n_changes_norm_t-54-40', 'number of changes 54-40', 'nchanges_54-40'),
        nbins=80,
        meanx=True, xmax=0.4, ymax=3.5)


plot_xy(('frac_new_users_vs_sw_t-0-0', 'fraction of new users 0-0', 'fracnewusers_0-0'),
        ('n_changes_norm_t-0-0', 'number of changes 0-0', 'nchanges_0-0'),
        nbins=200,
        meanx=True, meany=True, ymax=1.3)

plot_xy(('frac_new_users_vs_sw_t-0-0', 'fraction of new users 0-0', 'fracnewusers_0-0'),
        ('frac_attack_changes_t-0-0', 'fraction attack changes 0-0', 'fracattack'),
        nbins=200,
        meanx=True, meany=True)

plot_xy(('frac_new_users_vs_sw_t-0-0', 'fraction of new users 0-0', 'fracnewusers_0-0'),
        ('changes_per_user_sw_t-0-0', 'changes / user 0-0', 'changesperuser'),
        nbins=200,
        meanx=True, meany=True)


plot_1D(('frac_attack_changes_t-0-0', 'fraction attack changes 0-0', 'fracattack'))
plot_1D(('returnrate_t-0-0', 'return rate 0-0', 'returnrate'))

plot_xy(('frac_pixdiff_inst_vs_stable_t-84-70', 'fraction differing pixels inst 84-70', 'fracpixdiffinst_84-70'),
        ('n_changes_norm_t-84-70', 'number of changes 84-70', 'nchanges_84-70'),
        nbins=80,
        meanx=True, xmax=0.4, ymax=3.5)

plot_xy(('frac_pixdiff_inst_vs_stable_t-69-55', 'fraction differing pixels inst 69-55', 'fracpixdiffinst_69-55'),
        ('n_changes_norm_t-69-55', 'number of changes 69-55', 'nchanges_69-55'),
        nbins=80,
        meanx=True, xmax=0.4, ymax=3.5)

plot_xy(('frac_pixdiff_inst_vs_stable_t-54-40', 'fraction differing pixels inst 54-40', 'fracpixdiffinst_54-40'),
        ('frac_redundant_changes_t-54-40', 'fraction redundant changes 54-40', 'fracredun_54-40'),
        nbins=80,
        meanx=True, xmax=0.4)

plot_xy(('frac_pixdiff_inst_vs_stable_t-54-40', 'fraction differing pixels inst 54-40', 'fracpixdiffinst_54-40'),
        ('frac_redundant_changes_t-54-40', 'fraction redundant changes 54-40', 'fracredun_54-40'),
        nbins=80,
        meanx=True, xmax=0.4)


