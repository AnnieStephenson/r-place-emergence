import numpy as np
import xgboost as xg
import os, sys
import math
import scipy.stats
import pickle
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import seaborn as sns
import matplotlib.pyplot as plt
from rplacem import var as var
import gc
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import shap
import copy
import EvalML as eval
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.utils import shuffle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--param_num", default=0) # 0 is nominal result, 1 to 6 are sensitivity analysis
parser.add_argument("-t", "--test2023", default=False)
parser.add_argument("-s", "--shapplots", default=False)
parser.add_argument("-w", "--store_worstSHAP", default=False)
arg = parser.parse_args()
arg.shapplots = True if arg.store_worstSHAP else arg.shapplots

# GLOBAL AND XGBOOST PARAMETERS
ml_param = eval.AlgoParam(type='regression', # or 'classification
                          test2023=arg.test2023,
                          n_features=None,
                          num_rounds=None, 
                          learning_rate=0.035,    
                          max_depth=8, 
                          min_child_weight=None, 
                          subsample=0.8, 
                          colsample=0.75,
                          log_subtract_transform=1.5, 
                          weight_highEarliness=0.4,
                          calibrate_pred=True)
ml_param.num_rounds = 100 if ml_param.test2023 else 160 # Max number of boosting rounds (iterations)
ml_param.min_child_weight = 8 if ml_param.type == 'regression' else 4

warning_threshold_sec = 3600
warning_threshold = ml_param.transform_target(warning_threshold_sec)
makeplots = False if (ml_param.type != 'regression') else True
savefiles = True
corrplots = False # deactivates exclusion of worst time features
emax_test_hist2d = 1.3e3

file_excludedvars = os.path.join(var.DATA_PATH, 'excluded_variables_fromSHAP.txt')
exclude_timefeats = False#os.path.exists(file_excludedvars) if not corrplots else False
only_safetimemargin = False

param_str = var.param_str_fun(arg.param_num)
print(param_str)


# PLOTTING ROUTINES
def plot_training_vars(inputvals, outputval_orig, varnames, earliness_thres):
    '''
    Plot all training variables, separated in signal (earliness above given threshold) and background
    '''
    inds_sig = np.where(outputval_orig <= earliness_thres)
    inds_bkg = np.where(outputval_orig > earliness_thres)
    warn_thres_str = ( str(int(warning_threshold_sec/3600.))+'h' if warning_threshold_sec/3600. > 0.9 else str(int(warning_threshold_sec/60))+'min')

    plotted_vars = [0]
    for iv in range(len(varnames)):
        if varnames[iv][-3:] == '0-0':
            plotted_vars.append(iv)
    vals = inputvals[:, plotted_vars]
    vals[:, 0] = ml_param.transform_target(outputval_orig)

    nvars = len(plotted_vars)

    fig, axes = plt.subplots(int(nvars/4) + 1, 4, figsize=(10,8), num=1, clear=True)

    for var_idx, ax in enumerate(axes.T.flat):
        if var_idx < nvars:
            bins = 100 #np.logspace(np.log10(np.min(vals[:, var_idx])),np.log10(np.max(vals[:, var_idx])), 100)
            freq, edges = np.histogram(vals[inds_bkg, var_idx], bins=bins, density=True)
            bkgax, = ax.step(edges[:-1], freq, 'b', label='earliness > '+warn_thres_str)
            freq, edges = np.histogram(vals[inds_sig, var_idx], bins=bins, density=True)
            sigax, = ax.step(edges[:-1], freq, 'r', label='earliness <= '+warn_thres_str)
            ax.tick_params(axis='both', labelsize=7)
            ax.set_title((varnames[var_idx] if var_idx > 0 else 'earliness------')[:-6], fontsize=8)
            ax.set_yscale("log")
            #ax.set_yticks([])
        else:
            ax.remove()

        
    plt.subplots_adjust(hspace=0.5)
    fig.legend([sigax, bkgax], ['earliness <= '+warn_thres_str, 'earliness > '+warn_thres_str])
    fig.savefig(os.path.join(var.FIGS_PATH, 'ML', 'training_sample_variables.pdf'), dpi=350, bbox_inches='tight')


def scatter_true_vs_pred(true, pred, id_idx, test_inds, loge=True, onlysomecompos=False, trainsamples=False, addsavename=''):
    '''
    2D plot of earliness: true vs predicted
    '''
    fig = plt.figure(num=1, clear=True)
    sns.despine()

    plt.ylabel(r'100 + $\bf{predicted}$ earliness [s]', fontsize=12)
    plt.xlabel(r'100 + $\bf{true}$ earliness [s]', fontsize=12)
    emin = 100.
    emax = 2.8e5
    plt.xlim(emin, emax)
    plt.ylim(emin, emax)
    if loge:
        plt.xscale('log')
        plt.yscale('log')

    if onlysomecompos:
        select = (id_idx[test_inds]%10 == 1)
    else:
        select = np.arange(0, len(true))

    plt.scatter(100. + true[select][0:2000],
                100. + pred[select][0:2000],
                c=id_idx[test_inds][select][0:2000], 
                cmap='rainbow',
                s=1)
    
    plt.plot(np.linspace(emin,emax,10), np.linspace(emin,emax,10), color='black', linestyle='dashed', linewidth=0.5)
    plt.grid(color='black', linewidth=0.3, linestyle='dotted')

    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', ('log_' if loge else '') + 'earliness_pred_vs_true_scatter'+('_trainingsamples' if trainsamples else '')+addsavename+'.pdf'), 
                dpi=250, bbox_inches='tight')


def hist_true_vs_pred(true, pred, zmax=2e3, 
                      loge=True, 
                      trainsamples=False, normalize_cols=True, addsavename='',
                      addx=[], addy=[]):

    fig = plt.figure(num=1, clear=True)
    sns.despine()

    plt.ylabel(r'100 + $\bf{predicted}$ earliness [s]', fontsize=16)
    plt.xlabel(r'100 + $\bf{true}$ earliness [s]', fontsize=16)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    emin = 100.
    emax = 22*3600
    plt.xlim(emin, emax)
    plt.ylim(emin, emax)
    if loge:
        plt.xscale('log')
        plt.yscale('log')


    H, xedges, yedges = np.histogram2d( 100. + true,
                                        100. + pred,
                                        bins=[np.logspace(np.log10(emin), np.log10(emax), 40),
                                        np.logspace(np.log10(emin), np.log10(emax), 40)])
    H = H.T
    # normalize by columns
    if normalize_cols:
        H = H / np.sum(H, axis=0)
    X, Y = np.meshgrid(xedges, yedges)
    vm = max(1e-3 if normalize_cols else 1., np.nanmin(H[H > 0]))
    vM = min(np.nanmax(H), zmax)
    #H[(H < 1e-3) & (H > 3e-4)] = 1e-3
    #H[H < 3e-4] = np.NaN

    plt.pcolormesh(X, Y, H,  
                   cmap='inferno',
                   norm=colors.LogNorm(vmin=vm, vmax=vM) if loge else None
                   )
    
    '''
    plt.hist2d(100. + true,
               100. + pred,
               bins=[np.logspace(np.log10(emin), np.log10(emax), 50),
                     np.logspace(np.log10(emin), np.log10(emax), 50)],  
               cmap='inferno',
               norm=colors.LogNorm(vmin=0.9, vmax=zmax) if loge else None,
               )
    '''

    # additions for communication
    plt.text(0.92*1200, 125, r'$\bf{20 min}$', horizontalalignment='right', color='green', fontsize=15, bbox=dict(facecolor='white', edgecolor='white', boxstyle='square, pad=0.1', alpha=0.7))
    plt.text(0.92*3600, 125, r'$\bf{1 h}$', horizontalalignment='right', color='green', fontsize=15, bbox=dict(facecolor='white', edgecolor='white', boxstyle='square, pad=0.1', alpha=0.7))
    plt.text(0.92*3*3600, 125, r'$\bf{3 h}$', horizontalalignment='right', color='green', fontsize=15, bbox=dict(facecolor='white', edgecolor='white', boxstyle='square, pad=0.1', alpha=0.7))
    plt.text(0.93*6*3600, 130, r'$\bf{6 h}$', horizontalalignment='right', color='green', fontsize=14, bbox=dict(facecolor='white', edgecolor='white', boxstyle='square, pad=0.1', alpha=0.7))

    plt.plot(np.linspace(emin,emax,2), np.linspace(emin,emax,2), color='black', linestyle=':', linewidth=2)
    plt.hlines(y=[1200, 3600, 3*3600, 6*3600],
               xmin=[0,0,0,0], xmax=[1200, 3600, 3*3600, 6*3600], 
               linestyle='--', linewidth=1., color='green')
    plt.vlines(x=[1200, 3600, 3*3600, 6*3600],
               ymin=[0,0,0,0], ymax=[1200, 3600, 3*3600, 6*3600], 
               linestyle='--', linewidth=1., color='green')
    
    plt.colorbar(label=('probability (at given true earliness)' if normalize_cols else 'counts'), pad=0.02)

    #xminrect = 1.1*6*3600
    #plt.gca().add_patch(Rectangle((xminrect, 100), emax-xminrect, emax-100, alpha=0.75, color='white', linewidth=0))
    #plt.text(7e4, 1000, r'$\bf{not\ \ predictable}$', color='green', fontsize=14, rotation=90)

    plt.plot(100+np.array(addx),100+np.array(addy), color='blue')

    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', ('log_' if loge else '') + 'earliness_pred_vs_true_hist'+('_trainingsamples' if trainsamples else '')+addsavename+'.pdf'), 
                dpi=350, bbox_inches='tight')

def plot_features_importance(model, vnames, importance_type, imp_sort=False, log_imp=False):
    importance = model.get_score(importance_type=importance_type)
    fig, ax = plt.subplots(figsize=(6,14))
    for v in vnames:
        if v not in importance.keys():
            importance[v] = 0
    
    names = np.array(list(importance.keys()))
    imp = np.array(list(importance.values()))
    if imp_sort:
        sortidx = np.argsort(imp)
        imp = imp[sortidx]
        names = names[sortidx]
    plt.barh(names, imp)
    plt.xlabel = 'importance ('+importance_type+' method)'
    if log_imp:
        plt.xscale('log')
    plt.xlim([0.8*min(imp[imp != 0]), 1.2*max(imp)])
    plt.yticks(fontsize=5)
    fig.subplots_adjust(left=0.27, top=0.92, bottom=0.2, right=0.95)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'feature_importance_'+importance_type+('_sorted' if imp_sort else '')+'.pdf'), bbox_inches='tight')

def extremes_colormap(min=0):
    RdBu = plt.cm.get_cmap('RdBu', 400)
    cols = RdBu(np.linspace(0, 1, 400))
    colsred_condensed = cols[:200:(5 if min == 60 else (10 if min == 80 else 1))]
    colsblu_condensed = cols[200::(5 if min == 60 else (10 if min == 80 else 1))]
    white = np.array([1, 1, 1, 1])
    newcols = np.concatenate((colsred_condensed, np.full(((120 if min == 60 else (160 if min == 80 else 0)), 4), white), colsblu_condensed))
    newcmp = colors.ListedColormap(newcols)
    return newcmp

def correlation_heatmap(correlations, summary=False, zoom_extremes=True, vnames=None, annot=False, shap=False):
    fig= plt.subplots(figsize=(10,10), num=1, clear=True)
    sns.heatmap(correlations, vmax=1.0, vmin=-1.0, center=0, cmap=extremes_colormap(60 if zoom_extremes else 0), fmt='.2f', 
                square=True, linewidths=.0, annot=annot, annot_kws={"fontsize":3}, cbar_kws={"shrink": .70},
                xticklabels=vnames,
                yticklabels=vnames, 
                )
    plt.xticks(fontsize=10 if summary else 3)
    plt.yticks(fontsize=10 if summary else 3)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'correlation_matrix'+('_timevaraverage' if summary else '')+('_shap' if shap else '')+'.pdf'), bbox_inches='tight')

def plot_probas_BinaryClassification(sig, bkg, warn_thres_str, add_name, classif=True, logy=True):
    plt.figure(num=1, clear=True)
    emin = 0.95*np.min(sig)
    emax = 1 if classif else 1.05*max(np.max(bkg),np.max(sig))
    hsig = plt.hist(sig, bins=np.linspace(0,emax,350), histtype = 'step', label='true earliness < '+warn_thres_str, density=True)
    hbkg = plt.hist(bkg, bins=np.linspace(0,emax,350), histtype = 'step', label='true earliness > '+warn_thres_str, density=True)
    plt.legend()
    plt.xlabel('xbgoost output')
    if logy:
        plt.yscale('log')
    plt.ylabel('p.d.f.')
    plt.xlim([emin,emax])
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'proba_SigBkg_BinaryClassif_earlinessBelow'+warn_thres_str+add_name+'.png'), dpi=250)
    return hsig, hbkg

def SHAP_reject_messy_times(shapvals):
    vars_tocut = ['frac_pixdiff_inst_vs_swref_t-12-9',
                  'frac_pixdiff_inst_vs_swref_t-17-13',
                  'frac_pixdiff_inst_vs_swref_t-24-18',
                  'frac_pixdiff_inst_vs_swref_t-39-25'
                  ]
    thresholds = [0.1, 0.07, 0.07, 0.07]
    vars_idx = []
    for vtest in vars_tocut:
        for v in range(len(shapvals.feature_names)):
            if shapvals.feature_names[v] == vtest:
                vars_idx.append(v)
    print(shapvals.shape[0], 'instances entering SHAP_reject_messy_times')
    
    inds = np.where((shapvals.data[:, vars_idx[0]] < thresholds[0]) &
                    (shapvals.data[:, vars_idx[1]] < thresholds[1]) &
                    (shapvals.data[:, vars_idx[2]] < thresholds[2]) &
                    (shapvals.data[:, vars_idx[3]] < thresholds[3]))[0]
    print(len(inds), 'instances exiting SHAP_reject_messy_times')
    
    return inds

def reject_messy_times(inputvals, varnames):
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
    threshold = 0.35/6 #0.06
    vars_idx = []
    for vtest in vars_tocut:
        for v in range(len(varnames)):
            if varnames[v] == vtest:
                vars_idx.append(v)
    print(inputvals.shape[0], 'instances entering reject_messy_times')
    
    weights = np.array([1,1,2,2,3,4,5,7,11])
    vals_stabvars = inputvals[:, vars_idx]
    # for each instance, multiply weights and values of features element by element, then sum over features
    frac_pixdiff_sw = np.sum(np.multiply(weights, vals_stabvars), axis=1) / np.sum(weights)
    inds = np.where(frac_pixdiff_sw < threshold)[0]
    print(len(inds), 'instances exiting reject_messy_times')
    
    return inds

def SHAP_featureDependence(target, shapvals, targetsort, varname, varnum, ml_param, e_threshold=3600, onlyStableTimes=False):
    def xrange_lims(x):
        xlog = False
        xmin, xmax = np.min(x), np.max(x)
        #if (varname[0:7] == 'frac_at'):
        #    xmin = 1e-3
        #    x[x<xmin] = xmin
        #    xlog = True
        if (varname[0:7] == 'autocor'):
            xmin = -0.015 if xmin<-0.015 else xmin
            xmax = 0.015 if xmax>0.015 else xmax
            x[x<xmin] = xmin+1e-4
            x[x>xmax] = xmax-1e-4
        if (varname[0:7] == 'changes'):
            xmax = 7 if xmax>7 else xmax
            x[x>xmax] = xmax-1e-3
        if (varname[0:7] == 'cumul_a'):
            xmin = 2e-3 if xmin<2e-3 else xmin
            x[x<xmin] = 1.01*xmin
            xlog = True
        if (varname[0:7] == 'entropy'):
            xmax = 3.5 if xmax>3.5 else xmax
            x[x>xmax] = xmax-1e-3
        if (varname[0:23] == 'frac_pixdiff_inst_vs_st'):
            xmax = 0.25 if xmax>0.25 else xmax
            x[x>xmax] = xmax-1e-3
        if (varname[0:23] == 'frac_pixdiff_inst_vs_sw'):
            xmax = 0.45 if xmax>0.45 else xmax
            x[x>xmax] = xmax-1e-3
        if (varname[0:7] == 'fractal'):
            xmin = 0.65 if xmin<0.65 else xmin
            x[x<xmin] = xmin+1e-3
            xmax = 1.35 if xmax>1.35 else xmax
            x[x>xmax] = xmax-1e-3
        if (varname[0:7] == 'instabi'):
            xmin = 2e-3 if xmin<2e-3 else xmin
            x[x<xmin] = 1.01*xmin
            xmax = 0.2 if xmax>0.2 else xmax
            x[x>xmax] = 0.99*xmax
            xlog = True
        if (varname[0:7] == 'n_chang'):
            xmin = 1e-3 if xmin<1e-3 else xmin
            x[x<xmin] = 1.01*xmin
            xmax = 7 if xmax>7 else xmax
            x[x>xmax] = xmax-1e-3
            xlog = True
        if (varname[0:21] == 'n_used_colors_meantop'):
            xmax = 2.7 if xmax>2.7 else xmax
            x[x>xmax] = xmax-1e-3
        elif (varname[0:8] == 'n_used_c'):
            xmin = 0.99
            xmax = 1.5 if xmax>1.5 else xmax
            x[x>xmax] = xmax-1e-3
        if (varname[0:7] == 'n_users'):
            xmin = 1e-3 if xmin<1e-3 else xmin
            x[x<xmin] = 1.01*xmin
            xlog = True
        if (varname[0:7] == 'runneru'):
            xmin = 1e-2 if xmin<1e-2 else xmin
            x[x<xmin] = 1.01*xmin
            xlog = True
        #if (varname[0:12] == 'variance_all'):
        #    xmin = 0.9 if xmin<0.9 else xmin
        #    x[x<xmin] = xmin+1e-3
        #    xmax = 1.15 if xmax>1.15 else xmax
        #    x[x>xmax] = xmax-1e-3
        if (varname[0:16] == 'returntime_meant'):
            xmax = 5000 if xmax>5000 else xmax
            x[x>xmax] = 5000-1
        elif (varname[0:16] == 'returntime_mean_'):
            xmax = 3000 if xmax>3000 else xmax
            x[x>xmax] = 3000-1
        
        
        return xmin, xmax, xlog
    
    xmin, xmax, xlog = xrange_lims(shapvals.data[:, varnum])

    f = plt.figure(num=1, clear=True, figsize=(10,8))

    x = shapvals.data[targetsort, varnum]
    y = shapvals.values[targetsort, varnum]
    c = target[targetsort]
    sc = plt.scatter(x, y, s=2, c=c, cmap='viridis', #alpha=alpha, vmin=vmin, vmax=vmax
                )
    plt.xlim([xmin, xmax])
    if xlog:
        plt.xscale('log')
    cb = plt.colorbar(sc)
    cb.set_label('log(true time to transition)')
    plt.xlabel(varname)
    plt.ylabel('SHAP value for '+varname)
    plt.hlines(0, xmin, xmax, 'grey', 'dashed', linewidth=1.5)

    invtransf_targ = ml_param.invtransform_target(c)
    is_sig = np.where(invtransf_targ <= e_threshold)[0]
    is_bkg = np.where(invtransf_targ > e_threshold)[0]


    meanbkg = scipy.stats.binned_statistic(x[is_bkg], y[is_bkg], bins=(np.logspace(np.log10(xmin), np.log10(xmax), 120) if xlog else np.linspace(xmin, xmax, 120)), statistic='mean')
    mean_valbkg = meanbkg.statistic
    mean_binc_bkg = meanbkg.bin_edges
    mean_binc_bkg = (mean_binc_bkg[:-1] + mean_binc_bkg[1:])/2.
    plt.plot(mean_binc_bkg, mean_valbkg, color='orangered', label=r'mean SHAP (true $T^3 >$'+str(int(e_threshold))+'s)')

    mean = scipy.stats.binned_statistic(x[is_sig], y[is_sig], bins=(np.logspace(np.log10(xmin), np.log10(xmax), 100) if xlog else np.linspace(xmin, xmax, 100)), statistic='mean')
    mean_val = mean.statistic
    mean_binc = mean.bin_edges
    mean_binc = (mean_binc[:-1] + mean_binc[1:])/2.
    plt.plot(mean_binc, mean_val, color='violet', label=r'mean SHAP (true $T^3 \leq$'+str(int(e_threshold))+'s)')

    hist, b = np.histogram(x, bins=(np.logspace(np.log10(xmin), np.log10(xmax), 50) if xlog else np.linspace(xmin, xmax, 50)))
    ymin, ymax = np.min(y), np.max(y)
    hist = hist*(ymax-ymin)/np.sum(hist)
    plt.bar(b[:-1], hist, align='edge', width=np.diff(b), color='grey', alpha=0.2, label='pdf of feature', bottom=ymin)

    plt.legend(fontsize=9, framealpha=0.3)

    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP', 
                             'SHAP_scatter'+('_onlyStableTimes' if onlyStableTimes else '')+('_excludeworstSHAP' if exclude_timefeats else ''), 'SHAP_scatter_'+varname+'.png'), 
                             dpi=250, bbox_inches='tight')
    plt.clf()
    f.clear()





# CUSTOM OBJECTIVE AND EVALUATION FUNCTIONS

# Define the custom objective function
# reverse pred and dtrain args when using sklearn
def squaredrelerr(pred, dtrain): # objective fct = ((pred - true)/true)**2
    # could add another loss term for continuity of earliness prediction? ((pred(t) - pred(t-5min) + 5min) / constant(10?)*5min)**2
    # for this, need to add this continuity in the earliness values of no-transition compositions
    # and need to keep in [dtrain] the information about what event is pred(t-5min). 
    # But does this make sense on the test sample??
    # The test sample will actually not have this information, it is just used in the training to learn 'time ranking' from the usual input variables
    # maybe need to make the 5min more flexible at high earliness because log(earliness) is the predicted variable

    # This would just teach that an event happens after another given event? 
    # Isn't it contained already in the t-dependent input variables?
    # Yes but it is not an imperative that the algo considers that information so it might not be exploited
    true = dtrain.get_label()
    weight = dtrain.get_weight()
    #prev_tstep = dtrain.get_uint_info('previous_event_idx')
    grad = weight * (pred - true) / true**2
    hess = weight / true**2
    return grad, hess

# Define custom evaluation metrics
def meanrelerr(pred, dtrain):
    true = dtrain.get_label()
    weight = dtrain.get_weight()
    relative_errors = weight * np.abs((true - pred) / true)
    return 'meanrelerr', float(np.sum(relative_errors) / np.sum(weight))

def rmsrelerr(pred, dtrain):
    true = dtrain.get_label()
    weight = dtrain.get_weight()
    relerr2 = weight * np.power((true - pred) / true, 2)
    # neglect here the "N-1" that should be used in the denominator for unbiased rms
    return 'rmsrelerr', float(np.sqrt(np.sum(relerr2) / np.sum(weight)))

def negF1_20min(pred, dtrain):
    roctool = eval.EvalML(dtrain.get_label(), pred, true_threshold=ml_param.transform_target(1200), n_pred_cuts=70,
                          computeTN=False, transformed_earliness=True, algo_param=ml_param)
    roctool.set_F1()
    return 'F1_20min', -roctool.F1

def negF1_1h(pred, dtrain):
    roctool = eval.EvalML(dtrain.get_label(), pred, true_threshold=ml_param.transform_target(3600), n_pred_cuts=70,
                          computeTN=False, transformed_earliness=True, algo_param=ml_param)
    roctool.set_F1()
    return 'F1_1h', -roctool.F1

def negF1_3h(pred, dtrain):
    roctool = eval.EvalML(dtrain.get_label(), pred, true_threshold=ml_param.transform_target(3*3600), n_pred_cuts=70,
                          computeTN=False, transformed_earliness=True, algo_param=ml_param)
    roctool.set_F1()
    return 'F1_3h', -roctool.F1

def ROCAUC_1and3h_maxFPR0p2and1(pred, dtrain):
    roc20min = eval.EvalML(dtrain.get_label(), pred, true_threshold=ml_param.transform_target(1200), n_pred_cuts=70,
                           computeTN=False, transformed_earliness=True, algo_param=ml_param)
    roc1h = eval.EvalML(dtrain.get_label(), pred, true_threshold=ml_param.transform_target(3600), n_pred_cuts=70,
                        computeTN=False, transformed_earliness=True, algo_param=ml_param)
    roc3h = eval.EvalML(dtrain.get_label(), pred, true_threshold=ml_param.transform_target(3*3600), n_pred_cuts=70,
                        computeTN=False, transformed_earliness=True, algo_param=ml_param)
    roc20min_1 = -roc20min.calc_ROCAUC(maxFPR=1)
    roc1h_0p2 = -roc1h.calc_ROCAUC(maxFPR=0.2)
    roc1h_1 = -roc1h.calc_ROCAUC(maxFPR=1)
    roc3h_0p2 = -roc3h.calc_ROCAUC(maxFPR=0.2)
    pr1h = -roc1h.calc_PRAUC()
    pr3h = -roc3h.calc_PRAUC()
    return [('ROC20min', roc20min_1), ('ROC1h0p2', roc1h_0p2), ('ROC1h1', roc1h_1), ('PR1h', pr1h), ('ROC3h0p2', roc3h_0p2), ('PR3h', pr3h)]



# extract data from file
print('get input data')
olddata = False
file_path = os.path.join(var.DATA_PATH, 'training_data_'+(str)('401' if olddata else '584')+'variables_'+param_str+'.pickle') #3h-SW_widertimefromstart.pickle
with open(file_path, 'rb') as f:
    [inputvals, outputval, varnames, eventtime, id_idx, id_dict,
     coarse_timerange, 
     kendall_tau,
     n_traintimes, n_traintimes_coarse,
     safetimemargin
     ] = pickle.load(f) 
print(len(np.unique(id_idx)), 'unique compositions')
outputval = -outputval
if only_safetimemargin:
    outputval = outputval[np.all(safetimemargin, axis=1)] # satisfy both conditions of safetimemargin for each kept time instance
    inputvals = inputvals[np.all(safetimemargin, axis=1)] # satisfy both conditions of safetimemargin for each kept time instance
    eventtime = eventtime[np.all(safetimemargin, axis=1)] # satisfy both conditions of safetimemargin for each kept time instance
    id_idx = id_idx[np.all(safetimemargin, axis=1)] # satisfy both conditions of safetimemargin for each kept time instance

    
# extract data from 2023 file
if ml_param.test2023:
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','data','2023', 'training_data_389variables_'+param_str+'.pickle')
    with open(file_path, 'rb') as f:
        [inputvals_2023, outputval_2023, varnames, eventtime_2023, id_idx_2023, id_dict_2023,
        coarse_timerange, 
        kendall_tau,
        n_traintimes, n_traintimes_coarse,
        safetimemargin
        ] = pickle.load(f) 
    outputval_2023 = -outputval_2023
    print(len(np.unique(id_idx_2023)), 'unique compositions 2023')

# correct a bug
for iv in range(len(varnames)):
    if varnames[iv][0:14] == 'frac_users_new':
        inputvals[np.where(np.abs(inputvals[:, iv]) > 1)[0], iv] = 1
        if ml_param.test2023:
            inputvals_2023[np.where(np.abs(inputvals_2023[:, iv]) > 1)[0], iv] = 1

# INITIAL DATASETS
nmaxcomp = 1e5

# keep only certain variables
oldvars = False
# old variable selection [5,7,8,9,10,15,16,17,18,22,29,30,31,32,33]# old selection with old dataset [5,6,8,13,23,24,25,26,27,28]# best var selection [4,5,6,7,10,11,15,16,18,19,29,30,31,32,33]#[5,10,14,15,19,29,30,31,32,33]#[4,5,6,8,10,14,15,18,19,29,30,31,32,33]# 19 cumul, 4 instead of 5
if oldvars:
    vars_toremove = [4,6,7,8,10,11,15,16,18,19]+ list(range(29,51))  
else:
    vars_toremove = [4,6,7,8,10,11,15,16,18,19,   30,33,35,37,39,40,43,    45,46,47,48,49,50,      ]  # 3,22,      
vars_sparse = []#[14,22,3,36,17,31,38,5]#[3,22,  14,31,36,21,38,5]
vars_sparse = [i for i in vars_sparse if i not in vars_toremove] # remove elements that are in varse_toremove from vars_sparse
traintimes_sparse_noncoarse = ['84-70', '39-25', '8-6', '0-0']
traintimes_sparse_coarse = ['84-70', '39-25', '13-8', '0-0'] # coarse time ranges for sparse variables
sparse_timerange = np.array([i in vars_sparse for i in range(len(coarse_timerange))])

# exclude features with low SHAP values, from previous runs
use_oldfeats_list = False
if use_oldfeats_list:
    # use external list of feature names to keep
    file_keptfeatures = os.path.join(var.DATA_PATH, 'varnames_paper.txt')
    with open(file_keptfeatures, 'r') as f:
        features_tokeep = [l[:-1] for l in f.readlines()]
        print('keeping these features: ', features_tokeep, len(features_tokeep))
elif exclude_timefeats:
    with open(file_excludedvars, 'r') as f:
        features_toremove = [l[:-1] for l in f.readlines()]
        print('removing '+str(len(features_toremove))+' features: ', features_toremove)
else:
    features_toremove = []

# build the list of kept features
vars_touse = []
keptfeats = []
feat_varchange = [0]
kept_timeranges = [] # per variable, for later use for SHAP values
i_current = 0
for i_var in range(len(coarse_timerange)):
    timesteps_thisvar = n_traintimes_coarse if coarse_timerange[i_var] else n_traintimes 
    traintimes_sparse = traintimes_sparse_coarse if coarse_timerange[i_var] else traintimes_sparse_noncoarse
    if i_var not in vars_toremove:
        print(f"using variable {varnames[i_current].split('_t-')[0]:<40} with {len(traintimes_sparse_noncoarse) if sparse_timerange[i_var] else timesteps_thisvar:>3} timesteps")
        vars_toadd = []
        for i in range(i_current, i_current+timesteps_thisvar):
            add = False
            if use_oldfeats_list:
                add |= varnames[i] in features_tokeep
            else:
                if varnames[i] not in features_toremove:
                    add |= (not sparse_timerange[i_var]) or any(t in varnames[i] for t in traintimes_sparse)
            if add:
                vars_toadd.append(i)
            
        keptfeats += vars_toadd
        feat_varchange.append(len(keptfeats))
        vars_touse.append(i_var)
        kept_timeranges.append([i - i_current for i in vars_toadd])
    i_current += timesteps_thisvar
additional_vars = [i for i in range(len(varnames)-(5 if olddata else 8), len(varnames)) 
                   if (olddata or i not in ([len(varnames)-3, len(varnames)-2, len(varnames)-1] if oldvars else [len(varnames)-1 ])) and 
                                                                                (not use_oldfeats_list or varnames[i] in features_tokeep)]
n_addvars = len(additional_vars)
keptfeats += additional_vars

keptfeats = np.array(keptfeats)
ml_param.n_features = len(keptfeats)

#with open('data/2022/varnames_212.txt', 'w') as f:
#    for v in np.sort(varnames[keptfeats]):
#        f.write(v + '\n')

# SELECT KEPT VARIABLES
print('select kept variables')
stable_times_only = True
if stable_times_only:
    # only time instances with stable past sliding window
    stable_instances = reject_messy_times(inputvals, varnames)
    select = np.intersect1d(np.where(np.array(id_idx) < nmaxcomp)[0], stable_instances)
    if ml_param.test2023:
        stable_instances_2023 = reject_messy_times(inputvals_2023, varnames)
        select_2023 = np.intersect1d(np.where(np.array(id_idx_2023) < nmaxcomp)[0], stable_instances_2023)
else:
    select = np.where(np.array(id_idx) < nmaxcomp)[0]

outputval_orig = np.array(outputval[select])
outputval = ml_param.transform_target(outputval_orig)
inputvals = inputvals[select][:, keptfeats]
eventtime = eventtime[select]
inputvals = np.nan_to_num(inputvals)
if ml_param.test2023:
    outputval_orig_2023 = np.array(outputval_2023[select_2023])
    outputval_2023 = ml_param.transform_target(outputval_orig_2023)
    inputvals_2023 = inputvals_2023[select_2023][:, keptfeats]
varnames = varnames[keptfeats]
coarse_timerange = coarse_timerange[vars_touse]
sparse_timerange = sparse_timerange[vars_touse]

# extend inputvals and varnames with 10 dummy variables that only contain random numbers between 0 and 1
add_random_vars = False
if add_random_vars:
    n_random_vars = 20
    print('adding '+str(n_random_vars)+' random variables')
    random_vars = np.concatenate((np.random.rand(inputvals.shape[0], n_random_vars//2), 
                                  np.random.randn(inputvals.shape[0], n_random_vars//2)), axis=1)
    inputvals = np.concatenate((inputvals, random_vars), axis=1)
    for i in range(n_random_vars//2):
        varnames = np.append(varnames, 'rand_unif_'+str(i+1))
    for i in range(n_random_vars//2):
        varnames = np.append(varnames, 'rand_gaus_'+str(i+1))
    keptfeats = np.append(keptfeats, np.arange(len(varnames)-n_random_vars, len(varnames)))
else:
    n_random_vars = 0

id_idx = np.array(id_idx)[select]
ididx_unique, ididx_pointer2unique = np.unique(id_idx, return_inverse=True) #needed because there are compositions that are totally skipped
ncomp = len(ididx_unique)
n_events = len(outputval)
print('use',n_events,'events from',ncomp,'compositions for training (including 20 percent for testing when testing uses 2022)')
nvar = len(varnames)
print('use',nvar,'features')
if ml_param.test2023:
    id_idx_2023 = np.array(id_idx_2023[select_2023])
    ididx_unique_2023 = np.unique(id_idx_2023)
    ncomp_2023 = len(ididx_unique_2023) 
    n_events_2023 = len(outputval_2023)
    print('use',n_events_2023,'events from',ncomp_2023,'compositions for testing in 2023')


# EVENT WEIGHTS
weights = np.ones(n_events)
xbreak1 = 3*3600
xbreak2 = np.max(outputval_orig) # 12h
weights[outputval_orig >= xbreak2] = ml_param.weight_highEarliness
# power-law interpolation between these two limits
powerexp = math.log(1 / ml_param.weight_highEarliness) / math.log(xbreak1 / xbreak2)
powerlaw_inds = np.where((outputval_orig > xbreak1) & (outputval_orig < xbreak2))
weights[powerlaw_inds] = np.power(outputval_orig[powerlaw_inds] / xbreak1, powerexp)
if ml_param.test2023:
    weights_2023 = np.ones(n_events_2023)
    weights_2023[outputval_orig_2023 >= xbreak2] = ml_param.weight_highEarliness
    powerlaw_inds_2023 = np.where((outputval_orig_2023 > xbreak1) & (outputval_orig_2023 < xbreak2))
    weights_2023[powerlaw_inds_2023] = np.power(outputval_orig_2023[powerlaw_inds_2023] / xbreak1, powerexp)
weights[outputval_orig > 11.99*3600] *= 1.
weights[outputval_orig < 3600] *= 1.

# SPLIT THE DATA into training and testing sets
test_size = 0 if ml_param.test2023 else 0.2 # 20% of data for testing
if arg.param_num == 0 and olddata:
    valid_size = 0.1 if ml_param.test2023 else 0.2
    # need to split in terms of compositions and not events, so that there are no train-test correlations (ie events from the same compo both in train and test sample)
    seed=1
    np.random.seed(seed)
    print('seed',seed)
    shuf = np.random.permutation(ncomp)
    test_inds = np.where(shuf[ididx_pointer2unique] < test_size * ncomp)[0]
    train_inds = np.where(shuf[ididx_pointer2unique] >= test_size * ncomp)[0] # max(test_size, valid_size)
    #valid_inds = np.where(shuf[ididx_pointer2unique] < valid_size * ncomp)[0] # for now take validation sample from the test set, but should in principle be independent

    # store ids of test compos
    shufidx_intest = np.where(shuf < test_size * ncomp)[0]
    compidx_intest = ididx_unique[shufidx_intest]
    compid_intest = id_dict[compidx_intest]
    np.savez(os.path.join(var.DATA_PATH, 'test_compositions_IDs'+('_arXivVersion' if olddata else '')+'.npz'), compid_intest = compid_intest, allow_pickle=True)
    del shufidx_intest, compidx_intest, compid_intest
else:
    # grab with id_dict the compositions of the test sample determined with arg.param_num==0, and keep the test sample used in the arXiv published version, for reproducibility
    with np.load(os.path.join(var.DATA_PATH, 'test_compositions_IDs_arXivVersion.npz'), allow_pickle=True) as f:
        compid_intest = f['compid_intest']
    intest = np.isin(id_dict[id_idx], compid_intest)
    test_inds = np.where(intest)[0]
    train_inds = np.where(~intest)[0]

#print(np.sort(id_dict[np.unique(id_idx[test_inds])]), np.unique(id_idx[test_inds]).size )
    
print('#events in test sample', len(test_inds))
print('#events in train sample', len(train_inds))


# CREATE TRAIN AND TEST SAMPLES
X_train = inputvals[train_inds]
X_test = inputvals_2023 if ml_param.test2023 else inputvals[test_inds] 
#X_valid = inputvals[valid_inds]
w_train = weights[train_inds]
w_test = weights_2023 if ml_param.test2023 else weights[test_inds]
#w_valid = weights[valid_inds]
if ml_param.type == 'regression':
    y_train = outputval[train_inds]
    y_test = outputval_2023 if ml_param.test2023 else outputval[test_inds]
    #y_valid = outputval[valid_inds]
else:
    y_train = (outputval[train_inds] < warning_threshold).astype(int)
    y_test = ((outputval_2023 if ml_param.test2023 else outputval[test_inds]) < warning_threshold).astype(int)
    #y_valid = (outputval[valid_inds] < warning_threshold).astype(int)
del outputval, weights
gc.collect()

run_PCA = False
if run_PCA:
    # ========================================================================
    # SHARED PCA PREPROCESSING
    # ========================================================================
    print("\n" + "="*50)
    print("STEP 1: PCA PREPROCESSING")
    print("="*50)

    # Standardize features (crucial for PCA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compute weighted mean from training data
    w_pca = np.power(w_train, 1)  # weights for PCA, can be different from training weights
    weighted_mean = np.sum(X_train_scaled * w_pca.reshape(-1, 1), axis=0) / np.sum(w_pca)
    # Center data with weighted mean
    X_train_centered = X_train_scaled - weighted_mean
    X_test_centered = X_test_scaled - weighted_mean

    # Apply sqrt(weights) to centered training data
    X_train_weighted = X_train_centered * np.sqrt(w_pca).reshape(-1, 1)

    # Fit PCA on weighted training data
    pca = PCA(random_state=42)
    pca.fit(X_train_weighted)

    # Transform data (using original centered data, not weighted)
    X_train_pca_full = X_train_centered @ pca.components_.T
    X_test_pca_full = X_test_centered @ pca.components_.T


    ## Apply PCA
    #pca = PCA()#random_state=42)
    #X_train_pca_full = pca.fit_transform(X_train_scaled)
    #X_test_pca_full = pca.transform(X_test_scaled)

    # Determine optimal number of components (95% variance)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    n_components_99 = np.argmax(cumsum_var >= 0.99) + 1
    n_pca_comp = n_components_99#nvar

    print(f"Components for 95% variance: {n_components_95}")
    print(f"Components for 99% variance: {n_components_99}")
    print(f"Total explained variance (95%): {cumsum_var[n_components_95-1]:.4f}")

    # Use 95% variance components for modeling
    X_train_pca = X_train_pca_full[:, :n_pca_comp]
    X_test_pca = X_test_pca_full[:, :n_pca_comp]

    # Plot PCA variance explanation
    if makeplots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Individual component variance
        ax1.bar(range(1, len(cumsum_var)+1), 
                pca.explained_variance_ratio_[:len(cumsum_var)])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.set_yscale('log')
        # set y min to 1e-5
        ax1.set_ylim(1e-6, 0.2)
        ax1.grid(True, alpha=0.3)
        
        # Cumulative variance
        ax2.plot(range(1, len(cumsum_var)+1), cumsum_var[:len(cumsum_var)], 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax2.axhline(y=0.99, color='orange', linestyle='--', label='99% variance')
        ax2.axvline(x=n_components_95, color='r', linestyle=':', alpha=0.7)
        ax2.axvline(x=n_components_99, color='orange', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.set_ylim(0,1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'pca_variance_analysis.pdf'))

    # print feature names of ranked PCA components
    print("\nPCA Component Feature Names:")
    for i in range(len(cumsum_var)):
        feature_names = [f"{varnames[j]} ({pca.components_[i, j]:.2f})" for j in np.argsort(np.abs(pca.components_[i]))[::-1][:10]]
        print(f"Component {i+1}: {', '.join(feature_names)}")

    # PCA loadings plot
    if makeplots:
        print("Creating PCA loadings matrix plot...")
        pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.pcolor(pca_loadings, cmap='seismic', vmin=-np.max(np.abs(pca_loadings)), vmax=np.max(np.abs(pca_loadings)))
        ax.set_xticks(range(len(varnames)))
        ax.set_xticklabels(varnames, rotation=90, fontsize=8)
        ax.set_yticks(range(len(pca_loadings)))
        ax.set_yticklabels([f'PC{i+1}' for i in range(len(pca_loadings))])
        ax.set_title('PCA Loadings Matrix')
        ax.set_xlabel('Original Features')
        ax.set_ylabel('PCA Components')
        plt.colorbar(im, ax=ax, label='Loading Value')
        plt.tight_layout()
        plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'pca_loadings_matrix.pdf'))

use_pca_features = False and run_PCA
run_tSNE = True
use_tsne_features = True and run_tSNE
viz_tsne = False
if run_tSNE:
    # ========================================================================
    # PATHWAY 2: PCA → t-SNE → Visualization
    # ========================================================================
    print("\n" + "="*50)
    print("PATHWAY 2: PCA → t-SNE → Visualization")
    print("="*50)
    n_pca_comp = 170

    # Apply t-SNE on PCA features
    # Use subset for t-SNE if data is too large (t-SNE is computationally expensive)
    X_train_tsne_tmp = X_train_pca if use_pca_features else X_train
    # neutralize in X_train_tsne_tmp the features that have varnames == 'age' and 'canvas_quadrant'
    X_train_tsne_tmp[:, np.where(varnames == 'age')[0][0]] = np.random.rand(X_train_tsne_tmp.shape[0])
    X_train_tsne_tmp[:, np.where(varnames == 'canvas_quadrant')[0][0]] = np.random.rand(X_train_tsne_tmp.shape[0])
    n_tsne_samples = min(500000, len(X_train_tsne_tmp))
    tsne_indices = np.random.choice(len(X_train_tsne_tmp), n_tsne_samples, replace=False)

    X_train_tsne, y_train_tsne = X_train_tsne_tmp[tsne_indices], y_train[tsne_indices]
    del X_train_tsne_tmp

    print(f"Applying t-SNE on {n_tsne_samples} samples...")
    #umap.umap or tsne.TSNE
    tsne = umap.UMAP(n_components=2 if viz_tsne else n_pca_comp, #random_state=42, 
                     min_dist=0.5 if viz_tsne else 0.2, 
                     n_neighbors=90, 
                     #perplexity=50, #n_iter=1000, 
                     #init='random', learning_rate=10, n_jobs=8,
                verbose=True)
    X_tsne = tsne.fit_transform(X_train_tsne)
    y_tsne = y_train_tsne

    # t-SNE Visualization
    if makeplots and viz_tsne:
        print("Creating t-SNE visualizations...")
        cols = [(0, 'white'), (1, 'darkblue')]
        cmapblue = colors.LinearSegmentedColormap.from_list("white_to_darkblue", cols)

        fig, axes = plt.subplots(2, 2, figsize=(19, 12))
        
        # Basic t-SNE plot colored by target
        ax1 = axes[0, 0]
        if ml_param.type == 'regression':
            scatter = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne, 
                                    cmap='viridis', alpha=0.6, s=2)
            plt.colorbar(scatter, ax=ax1, label='Target Value')
        else:
            colors_discrete = ['blue', 'red']
            for i, label in enumerate(np.unique(y_tsne)):
                mask = y_tsne == label
                ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                            c=colors_discrete[i], label=f'Class {int(label)}', 
                            alpha=0.6, s=2)
            ax1.legend()
        
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.set_title('t-SNE Visualization - Colored by Target')
        ax1.grid(True, alpha=0.3)
        
        # Density plot
        ax2 = axes[0, 1]
        X_tsne = np.nan_to_num(X_tsne, nan=0.0, posinf=0.0, neginf=-0.0)
        ax2.hist2d(X_tsne[:, 0], X_tsne[:, 1], bins=50, cmap=cmapblue)
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.set_title('t-SNE Density Plot')

        # 2D plot of fraction of samples that have target below 3600, drawn in ax3
        ax3 = axes[1, 0]
        lowtarget = (y_tsne < ml_param.transform_target(3600)).astype(bool)
        # divide the hist2d for only lowtarget samples by that of all samples
        hist, xedges, yedges = np.histogram2d(X_tsne[:, 0], X_tsne[:, 1], bins=50)
        hist_lowtarget, _, _ = np.histogram2d(X_tsne[lowtarget, 0], X_tsne[lowtarget, 1], bins=(xedges, yedges))
        
        hist_lowtarget = np.nan_to_num(hist_lowtarget / hist, nan=0.0, posinf=0.0, neginf=-0.0)
        # plot this ratio
        im = ax3.pcolor(xedges, yedges, hist_lowtarget.T, cmap=cmapblue, shading='auto', vmin=0, vmax=0.05)
        plt.colorbar(im, ax=ax3, label='Fraction of Target < 3600s Samples')

        ax3.set_xlabel('t-SNE Component 1')
        ax3.set_ylabel('t-SNE Component 2')
        ax3.grid(True, alpha=0.3)

        # same with target below 3*3600s
        ax4 = axes[1, 1]
        lowtarget_3h = (y_tsne < ml_param.transform_target(3*3600)).astype(bool)
        hist_lowtarget_3h, _, _ = np.histogram2d(X_tsne[lowtarget_3h, 0], X_tsne[lowtarget_3h, 1], bins=(xedges, yedges))
        hist_lowtarget_3h = np.nan_to_num(hist_lowtarget_3h / hist, nan=0.0, posinf=0.0, neginf=-0.0)
        im_3h = ax4.pcolor(xedges, yedges, hist_lowtarget_3h.T, cmap=cmapblue, shading='auto', vmin=0, vmax=0.1)
        plt.colorbar(im_3h, ax=ax4, label='Fraction of Target < 3h Samples', cmap=cmapblue)
        ax4.set_xlabel('t-SNE Component 1')
        ax4.set_ylabel('t-SNE Component 2')
        ax4.grid(True, alpha=0.3)
        
        plt.ioff()
        plt.tight_layout()
        plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'tsne_analysis.pdf'), 
                        bbox_inches='tight', dpi=300)

        # ~3 plot with each 100 subplots. Each subplot shows the t-SNE points colored by one of the ~300 features
        n_features_to_plot = X_train_tsne.shape[1]
        n_subplots_per_fig = 25
        n_figs = int(np.ceil(n_features_to_plot / n_subplots_per_fig))
        for fig_idx in range(n_figs):
            start_idx = fig_idx * n_subplots_per_fig
            end_idx = min(start_idx + n_subplots_per_fig, n_features_to_plot)
            n_subplots = end_idx - start_idx
            n_cols = 5
            n_rows = int(np.ceil(n_subplots / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), num=999)
            axes = axes.flatten()
            for i, feature_idx in enumerate(range(start_idx, end_idx)):
                ax = axes[i]
                feature_values = np.nan_to_num(X_train_tsne[0:20000, feature_idx])
                # Compute 1st and 99th percentiles for colorbar limits
                vmin = np.percentile(feature_values, 1) if np.min(feature_values) != 0 else 0
                vmax = np.percentile(feature_values, 99)
                scatter = ax.scatter(X_tsne[0:20000, 0], X_tsne[0:20000, 1], c=feature_values, 
                            cmap='rainbow', alpha=0.6, s=1, vmin=vmin, vmax=vmax)
                plt.colorbar(scatter, ax=ax, label=f'{varnames[feature_idx]}')
                ax.set_title(f'{varnames[feature_idx]}')
                ax.set_xlabel('t-SNE Component 1')
                ax.set_ylabel('t-SNE Component 2')
                ax.grid(True, alpha=0.3)
            # remove empty subplots
            for j in range(n_subplots, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            plt.savefig(os.path.join(var.FIGS_PATH, 'ML', f'tsne_features_analysis_feats{start_idx}_to_{end_idx}.pdf'), 
                bbox_inches='tight', dpi=300)
            plt.close('all')
            fig.clear()
            del fig, axes, ax, scatter
            gc.collect()

    # recover age and canvas_quadrant features in the t-SNE transformed data. Just replace the random numbers with the original values
    if use_tsne_features:  
        X_train_tsne[:, np.where(varnames == 'age')[0][0]] = X_train[tsne_indices, np.where(varnames == 'age')[0][0]]
        X_train_tsne[:, np.where(varnames == 'canvas_quadrant')[0][0]] = X_train[tsne_indices, np.where(varnames == 'canvas_quadrant')[0][0]]

    #sys.exit()






# ========================================================================
# PATHWAY 1: PCA → XGBoost → SHAP
# ========================================================================
print("\n" + "="*50)
print("PATHWAY 1: PCA → XGBoost → SHAP")
print("="*50)

# Create PCA component names for interpretability
pca_feature_names = [f'PC{i+1}' for i in range(n_pca_comp)]

# DMatrix input for XGBoost
# no weights for binary classification
print('start dtrain')
dtrain = xg.DMatrix(data = X_train_pca if use_pca_features else tsne.transform(X_train) if use_tsne_features else X_train, 
                    label = y_train, weight=(w_train if ml_param.type == 'regression' else None), 
                    feature_names=pca_feature_names if use_pca_features or use_tsne_features else varnames)
print('start dtest')
dtest = xg.DMatrix(data = X_test_pca if use_pca_features else tsne.transform(X_test) if use_tsne_features else X_test, 
                   label = y_test, weight=(w_test if ml_param.type == 'regression' else None),  
                   feature_names=pca_feature_names if use_pca_features or use_tsne_features else varnames)
#dvalid = xg.DMatrix(data = X_valid, label = y_valid, weight=(w_valid if ml_param.type == 'regression' else None), feature_names=varnames)

del w_train, w_test
if ml_param.test2023:
    del outputval_2023, weights_2023, outputval_orig_2023, inputvals_2023
gc.collect()

if makeplots and corrplots:
    # plot correlation between variables
    print('plot correlations')

    a_forcorrs = np.hstack((X_train,y_train[:,np.newaxis]))
    corrs = np.corrcoef(a_forcorrs, rowvar=False)
    # correlation matrix for features
    correlation_heatmap(corrs, 
                        zoom_extremes=True, vnames=varnames)
    
    # correlation matrix for variables
    corrmat_pervar = np.zeros((len(feat_varchange)+n_addvars, len(feat_varchange)+n_addvars))

    vnames_eachvar = []
    for i_var in range(len(feat_varchange)+n_addvars):
        ivar_fullbkw = i_var-len(feat_varchange)-n_addvars+1
        vname = varnames[feat_varchange[i_var]].split('_t-')[0] if i_var < len(feat_varchange)-1 else (varnames[ivar_fullbkw] if ivar_fullbkw < 0 else 'time-to-transition')
        vnames_eachvar.append(vname)

        for j_var in range(len(feat_varchange)+n_addvars):
            jvar_fullbkw = j_var-len(feat_varchange)-n_addvars+1
            i_tokeep, j_tokeep = [], []

            # handle correlations with time-to-trans
            if i_var >= len(feat_varchange) - 1 and j_var >= len(feat_varchange) - 1:
                corrmat_pervar[i_var, j_var] = corrs[ivar_fullbkw-1, jvar_fullbkw-1]
            else:
                if i_var >= len(feat_varchange) - 1:
                    i_tokeep, j_tokeep = ivar_fullbkw-1, range(feat_varchange[j_var], feat_varchange[j_var+1])
                elif j_var >= len(feat_varchange) - 1:
                    i_tokeep, j_tokeep = range(feat_varchange[i_var], feat_varchange[i_var+1]), jvar_fullbkw-1
                else:
                    for i in range(feat_varchange[i_var], feat_varchange[i_var+1]):
                        feattime = varnames[i].split('_t-')[1]
                        try: # test if this time feature is present in the j list of time features
                            feattime_for_varj = [varnames[j].split('_t-')[1] for j in range(feat_varchange[j_var], feat_varchange[j_var+1])].index(feattime)
                            i_tokeep.append(i)
                            j_tokeep.append(feat_varchange[j_var] + feattime_for_varj)
                        except:
                            pass

                correlations_tosum = corrs[np.array(i_tokeep), np.array(j_tokeep)]
                corrmat_pervar[i_var, j_var] = np.mean(correlations_tosum)

    reindex = np.arange(corrmat_pervar.shape[0])
    correlation_heatmap(corrmat_pervar[reindex][:, reindex], summary=True, zoom_extremes=False, vnames=vnames_eachvar, annot=True)
    with open(os.path.join(var.DATA_PATH, 'correlation_matrix.pickle'), 'wb') as f:
        pickle.dump([corrmat_pervar, vnames_eachvar], f, protocol=pickle.HIGHEST_PROTOCOL)
    #print('plot training variables')
    #plot_training_vars(inputvals=inputvals, outputval_orig=outputval_orig, varnames=varnames, earliness_thres=warning_threshold_sec)
    del a_forcorrs

del X_train
if not ml_param.test2023:
    del inputvals, outputval_orig#, X_valid
gc.collect()

if corrplots:
    # plot all variables at t_0-0 versus time-to-transition
    ncol = 8
    fig, axis = plt.subplots(6, ncol, figsize=(7, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    fig.text(0.06, 0.5, 'log(100 + time-to-transition) - 1.5', ha='center', va='center', rotation='vertical', fontsize=16)
    iaxis=0
    for v in range(len(varnames)):
        if not 't-0-0' in varnames[v] and 't-' in varnames[v]:
            continue

        ax = axis[int(iaxis)//ncol, int(iaxis)%ncol] 
        xlimits = np.percentile(X_test[:,v], [0.5, 99.5 if (varnames[v]!='log(area)') else 99.9])
        xlimits += np.array([-0.03, 0.03]) * (xlimits[1]-xlimits[0])
        nbins=50
        ax.hist2d(X_test[:,v],y_test,     bins=nbins,             norm=colors.LogNorm()     ,range=[xlimits, [0.5,3.2]])
        ax.set_xlim(xlimits)
        ax.set_ylim(0.5,3.2)
        ax.set_title(varnames[v][:-6] if 't-0-0' in varnames[v] else varnames[v], fontsize=8)

        xmean = scipy.stats.binned_statistic(y_test, X_test[:,v], bins=nbins, range=[0.5,3.2], statistic='mean')
        xmeanerr = scipy.stats.binned_statistic(y_test, X_test[:,v], bins=nbins, range=[0.5,3.2], statistic=scipy.stats.sem)
        ax.plot(xmean.statistic,(xmean.bin_edges[1:] + xmean.bin_edges[:-1])/2, color='red')
        ax.fill_betweenx((xmean.bin_edges[1:] + xmean.bin_edges[:-1])/2, xmean.statistic - xmeanerr.statistic, xmean.statistic + xmeanerr.statistic, edgecolor='none', facecolor='red', alpha=0.5)
        iaxis += 1
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML','all_variables_0-0_vs_earliness.pdf'))

# XGBOOST REGRESSOR PARAMETERS
params = {
    'max_depth': ml_param.max_depth,                    # Maximum depth of each tree
    'learning_rate': ml_param.learning_rate,              # Learning rate
    'subsample': ml_param.subsample,                  # Subsample ratio of the training instances
    'colsample_bytree': ml_param.colsample,           # Subsample ratio of features at each new tree
    'colsample_bylevel': ml_param.colsample,          # Subsample ratio of features at each tree level
    'min_child_weight': ml_param.min_child_weight,             # minimum sum of event weights (times the hessian value) per node
    'random_state': 42,                # Random seed
    #'verbosity' : 2
}
if ml_param.type != 'regression':
    params['objective'] = 'binary:logistic'          # Objective function
    params['eval_metric'] = ['auc', 'logloss']


# TRAIN THE MODEL
eval_res = {}
print('start training, with',len(train_inds),'events')
model = xg.train(params, dtrain, 
                num_boost_round=ml_param.num_rounds,
                early_stopping_rounds=ml_param.num_rounds,
                obj=(squaredrelerr if ml_param.type == 'regression' else None),
                custom_metric=(ROCAUC_1and3h_maxFPR0p2and1 if ml_param.type == 'regression' else None),
                evals=[(dtest, 'test'), (dtrain, 'train')],
                evals_result=eval_res
                )
print('done training')  


# EVALUATE THE MODEL
# weights are in general ignored here
predictions = model.predict(dtest)
predictions_train = model.predict(dtrain)
#predictions_valid = model.predict(dvalid)
del dtrain, dtest

# return earliness from its transformation used in training
true_test = ml_param.invtransform_target(y_test) if ml_param.type == 'regression' else y_test
pred_test = ml_param.invtransform_target(predictions) if ml_param.type == 'regression' else predictions
true_train = ml_param.invtransform_target(y_train) if ml_param.type == 'regression' else y_train
pred_train = ml_param.invtransform_target(predictions_train) if ml_param.type == 'regression' else predictions_train

# CALIBRATE THE MODEL OUTPUT (PREDICTIONS) WITH VALIDATION SET
# calibration done on transformed (log) earliness
if ml_param.type == 'regression' and ml_param.calibrate_pred:

    keepinds = np.array(pred_test < 3600 + 3.5*true_test)
    kept_y = y_test[keepinds]
    kept_pred = predictions[keepinds]
    nonkept_y = y_test[~keepinds]
    nonkept_pred = predictions[~keepinds]

    # median prediction at each true value
    pred_valid_calib_binstats = scipy.stats.binned_statistic(kept_y, kept_pred,
                                                             bins=np.linspace(np.min(kept_y), np.max(kept_y)*1.01, 45), statistic='median')

    pred_medianpred = pred_valid_calib_binstats.statistic
    y_medianpred = pred_valid_calib_binstats.bin_edges
    y_medianpred = (y_medianpred[:-1] + y_medianpred[1:])/2.
    # remove nan's
    y_medianpred = np.delete(y_medianpred, np.where(np.isnan(pred_medianpred))[0])
    pred_medianpred = np.delete(pred_medianpred, np.where(np.isnan(pred_medianpred))[0])

    # linear fit
    def fitfct(x, xbreak, ybreak, xfin, yfin, ymin):
        offset = 2 - ml_param.log_subtract_transform
        xnew = x - offset
        xbreaknew = ml_param.transform_target(xbreak) - offset
        ybreaknew = ml_param.transform_target(ybreak) - offset
        yminnew = ml_param.transform_target(ymin) - offset
        xfinnew = ml_param.transform_target(xfin) - offset
        yfinnew = ml_param.transform_target(yfin) - offset - yminnew

        y = np.zeros(x.shape)
        y[xnew < xbreaknew] = yminnew + ybreaknew * xnew[xnew < xbreaknew] / xbreaknew
        slope2 = (yfinnew-ybreaknew) / (xfinnew-xbreaknew)
        y[xnew >= xbreaknew] = yminnew + ybreaknew + slope2 * (xnew[xnew >= xbreaknew] - xbreaknew)
        return y + offset

    maxtruecalib = 20*3600
    def fitfct_constr(x, xbreak, ybreak):
        return fitfct(x, xbreak, ybreak, xfin=maxtruecalib, 
                      yfin=np.max(ml_param.invtransform_target(kept_pred)), 
                      ymin=np.min(ml_param.invtransform_target(kept_pred)))
    
    # actual fit
    popt, pcov = curve_fit(fitfct_constr, y_medianpred, pred_medianpred, 
                           bounds=([0.3*3600, 0.5*3600],
                                   [5*3600, 4*3600]))

    y_valid_bincenters = np.linspace(ml_param.transform_target(0), ml_param.transform_target(maxtruecalib), 100)
    pred_valid_calib_vals = fitfct_constr(y_valid_bincenters, *popt)
    print('calib fit parameters: ', popt)

    if makeplots:
        fig = plt.figure(num=1, clear=True)
        plt.ylabel(r'100 + $\bf{predicted}$ time-to-transition [s]', fontsize=12)
        plt.xlabel(r'100 + $\bf{true}$ time-to-transition [s]', fontsize=12)
        emin = 100.
        emax = 20*3600
        plt.xlim(emin, emax)
        plt.ylim(emin, emax)
        plt.xscale('log')
        plt.yscale('log')

        plt.scatter(100+ml_param.invtransform_target(kept_y),
                    100+ml_param.invtransform_target(kept_pred), s=0.1, label='kept for calibration')
        plt.scatter(100+ml_param.invtransform_target(nonkept_y),
                    100+ml_param.invtransform_target(nonkept_pred), s=0.1, c='red', label='rejected for calibration')

        plt.plot(np.linspace(emin,emax,10), np.linspace(emin,emax,10), color='black', linestyle='dashed', linewidth=0.5)
        plt.plot(100+ml_param.invtransform_target(y_valid_bincenters), 100+ml_param.invtransform_target(pred_valid_calib_vals), color='black', label='calibration fit')
        plt.plot(100+ml_param.invtransform_target(y_medianpred), 100+ml_param.invtransform_target(pred_medianpred), color='green', label='median predicted')
        plt.grid(color='black', linewidth=0.3, linestyle='dotted')
        plt.legend(loc='lower right')

        plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'log_earliness_pred_vs_true_scatter'+('_test2023' if ml_param.test2023 else '')+('_excludeworstSHAP' if exclude_timefeats else '')+'.png'), 
                    dpi=250, bbox_inches='tight')

        # show calibration vs uncalibrated true vs predicted 2D histo
        hist_true_vs_pred(true_test, pred_test, zmax=emax_test_hist2d, 
                        loge=True, normalize_cols=True, addsavename='_highEweight'+str(ml_param.weight_highEarliness)+('_test2023' if ml_param.test2023 else '')+'_showCalibration',
                        addx=ml_param.invtransform_target(y_valid_bincenters), addy=ml_param.invtransform_target(pred_valid_calib_vals))
    
    # transform the predictions with these fit results
    predictions_calib = np.interp(predictions, pred_valid_calib_vals, y_valid_bincenters)
    predictions_train_calib = np.interp(predictions_train, pred_valid_calib_vals, y_valid_bincenters)
    pred_test = ml_param.invtransform_target(predictions_calib) if ml_param.type == 'regression' else predictions
    pred_train = ml_param.invtransform_target(predictions_train) if ml_param.type == 'regression' else predictions








if ml_param.type != 'regression':
    print('final auc train/test = ', eval_res['train']['auc'][-1] / eval_res['test']['auc'][-1])

# Make EvalML results for all tested earliness thresholds
evals_allthresholds = []
for warn_thres_sec in ([1200, 3600, 3*3600, 6*3600] if ml_param.type == 'regression' else [warning_threshold_sec]):

    eval_all = eval.EvalML(true_test, pred_test, true_threshold=warn_thres_sec, trainsample=False, algo_param=ml_param, transformed_earliness=False)
    eval_all.plotROC()
    eval_all.plotPR()
    eval_all.printResults()
    evals_allthresholds.append(eval_all)

    # make EvalML result for ratio of train to test samples
    eval_all_train = eval.EvalML(true_train, pred_train, pred_thresholds=eval_all.pred_thresholds, true_threshold=warn_thres_sec, algo_param=ml_param, transformed_earliness=False, trainsample=True )
    eval_ratio = eval.ratioEvals(eval_all_train, eval_all, printmore=True)
    eval_ratio.variableName = 'TrainOverTest'
    evals_allthresholds.append(eval_ratio)

    # plot distributions of predicted output
    warn_thres_str = ( str(int(warn_thres_sec/3600.))+'h' if warn_thres_sec/3600. > 0.9 else str(int(warn_thres_sec/60))+'min')
    is_warning = (true_test < warn_thres_sec) if ml_param.type == 'regression' else (true_test > 0.5)
    plot_probas_BinaryClassification(predictions[is_warning], predictions[np.invert(is_warning)], warn_thres_str, ('_regression' if ml_param.type == 'regression' else '_classification'), classif=(ml_param.type != 'regression'))


# SAVE EVALML results
if savefiles:
    with open(os.path.join(var.DATA_PATH, 'MLevaluation'
                                        +'_'+ ('regression' if ml_param.type == 'regression' else 'classification')
                                        +('_test2023' if ml_param.test2023 else '')
                                        +(('_earlinessBelow'+warn_thres_str) if ml_param.type != 'regression' else '')
                                        +'_maxdepth'+str(ml_param.max_depth)
                                        +'_minchildweight'+str(ml_param.min_child_weight)
                                        +'_subsample'+str(ml_param.subsample)
                                        +'_colsample'+str(ml_param.colsample)
                                        +'_logsubtransform'+str(ml_param.log_subtract_transform)
                                        +'_'+param_str
                                        +('_stabletimes' if stable_times_only else '')
                                        +'.pickle'), 'wb') as f:
        pickle.dump(evals_allthresholds, f, protocol=pickle.HIGHEST_PROTOCOL)

for e in evals_allthresholds:
    del e
del evals_allthresholds

# SAVE REGRESSION RESULT FOR ALL TEST EVENTS     
if savefiles:
    print('save to npz')
    np.savez(os.path.join(var.DATA_PATH, 'earliness_true_vs_predicted_'+ ('regression' if ml_param.type == 'regression' else 'classification')+('_test2023' if ml_param.test2023 else '')+'_'+param_str+('_stabletimes' if stable_times_only else '')+'.npz'),
                        true = true_test, predicted = pred_test, compoID_idx = id_idx[test_inds], time = eventtime[test_inds], id_dict = id_dict )

# PLOTS
if makeplots or arg.shapplots:
    print('first 2D histogram')
    # 2D true vs predicted histograms
    #scatter_true_vs_pred(true_test, pred_test, id_idx, test_inds, loge=True)
    #scatter_true_vs_pred(true_test, pred_test, id_idx, test_inds, loge=False)
    hist_true_vs_pred(true_test, pred_test, zmax=emax_test_hist2d, 
                    loge=True, normalize_cols=True, addsavename='_highEweight'+str(ml_param.weight_highEarliness)+('_test2023' if ml_param.test2023 else '')+('_excludeworstSHAP' if exclude_timefeats else ''))
    print('another 2D histogram')
    # check for overtraining with true vs pred_test for training sample
    #scatter_true_vs_pred(true_train, pred_train, id_idx, train_inds, loge=True, trainsamples=True)
    hist_true_vs_pred(true_train, pred_train, zmax=emax_test_hist2d*(3 if ml_param.test2023 else (1-test_size)/test_size), 
                    loge=True, trainsamples=True, normalize_cols=True, addsavename='_highEweight'+str(ml_param.weight_highEarliness)+('_test2023' if ml_param.test2023 else '')+('_excludeworstSHAP' if exclude_timefeats else ''))

    if arg.shapplots:
        del y_train, true_test, true_train, pred_test, pred_train, id_idx, eventtime
        if ml_param.test2023:
            del id_idx_2023, eventtime_2023
        gc.collect()
        print('start shap values')
        # SHAP values for feature importance
        explainer = shap.Explainer(model, feature_names=varnames)
        shap_values = explainer(X_test)

        print('aggregate shap')
        # Aggregate SHAP values by variable or by timerange
        shap_eachvar = copy.deepcopy(shap_values)
        shap_eachrange = copy.deepcopy(shap_values)
        shap_eachrange_coarse = copy.deepcopy(shap_values)

        nvar_tintegrated = len(feat_varchange)-1
        varmap_eachvar = []
        varmap_eachrange_coarse, varmap_eachrange = [], []
        for i in range(n_traintimes_coarse):
            varmap_eachrange_coarse.append([])
        for i in range(n_traintimes):
            varmap_eachrange.append([])
        varnames_eachvar = []
        varnames_eachrange = np.empty(n_traintimes, dtype=object)
        varnames_eachrange_coarse = np.empty(n_traintimes_coarse, dtype=object)

        featidx = 0
        v_coarse = 0
        v_noncoarse = 0
        for v in range(nvar_tintegrated):
            varmap_eachvar.append(np.arange(feat_varchange[v], feat_varchange[v+1]))
            varnames_eachvar.append(varnames[feat_varchange[v]].split('_t-')[0])

            for iconsec, isparse in enumerate(kept_timeranges[v]):
                if coarse_timerange[v]:
                    varmap_eachrange_coarse[isparse].append(feat_varchange[v] + iconsec)
                    varnames_eachrange_coarse[isparse] = 'time' + varnames[feat_varchange[v] + iconsec].split('_t-')[1]
                else:
                    varmap_eachrange[isparse].append(feat_varchange[v] + iconsec)
                    varnames_eachrange[isparse] = 'time' + varnames[feat_varchange[v] + iconsec].split('_t-')[1]

        print('end aggregate shap. Compute SHAP per variable')
        # New SHAP values for each variable
        shap_eachvar_vals = np.empty((shap_values.shape[0], nvar_tintegrated))
        shap_eachvar_data = np.empty((shap_values.shape[0], nvar_tintegrated))
        for v in range(nvar_tintegrated):
            shap_eachvar_vals[:, v] = np.sum(shap_values.values[:,varmap_eachvar[v]], axis=1)
            shap_eachvar_data[:, v] = np.mean(shap_values.data[:,varmap_eachvar[v]], axis=1)
        shap_eachvar.values = np.hstack((shap_eachvar_vals, shap_values.values[:, -n_addvars:]))
        shap_eachvar.data = np.hstack((shap_eachvar_data, shap_values.data[:, -n_addvars:]))
        shap_eachvar.feature_names = varnames_eachvar+shap_values.feature_names[-n_addvars:]

        print('Compute SHAP per timerange')
        # New SHAP values for each time range
        shap_eachrange_vals = np.empty((shap_values.shape[0], n_traintimes))
        shap_eachrange_data = np.empty((shap_values.shape[0], n_traintimes))
        for v in range(n_traintimes):
            shap_eachrange_vals[:, v] = np.sum(shap_values.values[:,varmap_eachrange[v]], axis=1)
            shap_eachrange_data[:, v] = np.mean(shap_values.data[:,varmap_eachrange[v]], axis=1) # this is not correct but is only used to show the feature value in the plots
        shap_eachrange.values = shap_eachrange_vals
        shap_eachrange.data = shap_eachrange_data
        shap_eachrange.feature_names = list(varnames_eachrange)

        # New SHAP values for each coarse time range
        shap_eachrange_coarse_vals = np.empty((shap_values.shape[0], n_traintimes_coarse))
        shap_eachrange_coarse_data = np.empty((shap_values.shape[0], n_traintimes_coarse))
        for v in range(n_traintimes_coarse):
            shap_eachrange_coarse_vals[:, v] = np.sum(shap_values.values[:,varmap_eachrange_coarse[v]], axis=1)
            shap_eachrange_coarse_data[:, v] = np.mean(shap_values.data[:,varmap_eachrange_coarse[v]], axis=1) # this is not correct but is only used to show the feature value in the plots
        shap_eachrange_coarse.values = shap_eachrange_coarse_vals
        shap_eachrange_coarse.data = shap_eachrange_coarse_data
        shap_eachrange_coarse.feature_names = list(varnames_eachrange_coarse)

        inds_onlyStableTimes = np.arange(shap_values.values.shape[0])#SHAP_reject_messy_times(shap_values)

        # SAVE SHAP VALUES
        if savefiles:
            with open(os.path.join(var.DATA_PATH, 'SHAP_values'+('_excludeworstSHAP' if exclude_timefeats else '')+'_'+param_str+('_stabletimes' if stable_times_only else '')+'.pickle'), 'wb') as f:
                pickle.dump([shap_values, shap_eachvar, shap_eachrange, shap_eachrange_coarse, inds_onlyStableTimes, ml_param],
                            f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # get the correlation matrix of the SHAP values
        print('plot correlations of SHAP values')
        shap_corrs = np.corrcoef(shap_values.values, rowvar=False)
        # correlation matrix for features
        correlation_heatmap(shap_corrs, shap=True,
                            zoom_extremes=False, vnames=shap_values.feature_names)
            
        # get the 10 or 20 least performing (and not too correlated) features
        n_shaptolook = min(250 + n_random_vars, len(varnames))
        n_toexclude = 20
        lowshap_highcorr = True # if True, then exclude features with low SHAP and high correlation with kept features
        no_corr_with_excluded = True # if True, then do not exclude features that are very correlated with other rejected features
        lowershap_than_rdm = True # if True, then exclude features with lower SHAP than the maximum SHAP of the 'rand_' random features
        if arg.store_worstSHAP:
            shap_meanabs_perfeature = np.mean(np.abs(shap_values.values), axis=0)
            feature_idx_sortedshap = np.argsort(shap_meanabs_perfeature)
            feat_idx_mayexclude = feature_idx_sortedshap[0:n_shaptolook]
            feat_idx_keep = feature_idx_sortedshap[n_toexclude:]

            corr_badSHAP = np.corrcoef(X_test[:, feat_idx_mayexclude], rowvar=False)
            corr_cross = np.corrcoef(X_test[:, feat_idx_mayexclude], X_test[:, feature_idx_sortedshap[n_shaptolook:]], rowvar=False)
            np.fill_diagonal(corr_cross, 0)
            np.fill_diagonal(shap_corrs, 0)
            mayexclude = np.arange(n_shaptolook)
           
            # Get argmax indices and max values and feature names for the correlation matrices with kept features
            # correlation between feature values
            corr_matrix_with_kept = np.abs(corr_cross[np.ix_(np.arange(n_shaptolook), np.arange(n_toexclude, corr_cross.shape[1]))])
            corr_argmax_indices = np.argmax(corr_matrix_with_kept, axis=1)
            max_corr_with_keep = np.max(corr_matrix_with_kept, axis=1)
            max_corr_featname = [varnames[feat_idx_keep[corr_argmax_idx]] for corr_argmax_idx in corr_argmax_indices]
            
            # correlation between shap values of features
            shap_corr_matrix_with_kept = np.abs(shap_corrs[np.ix_(feat_idx_mayexclude, feat_idx_keep)])
            shap_argmax_indices = np.argmax(shap_corr_matrix_with_kept, axis=1)
            max_shapcorr_with_keep = np.max(shap_corr_matrix_with_kept, axis=1)
            max_shapcorr_featname = [varnames[feat_idx_keep[shap_argmax_idx]] for shap_argmax_idx in shap_argmax_indices]
            
            # Compute shap_corr/(1-feature_corr) ratio and find the feature with the highest ratio
            ratio_corr_matrix = shap_corr_matrix_with_kept / (1.01 - corr_matrix_with_kept) 
            ratio_argmax_indices = np.argmax(ratio_corr_matrix, axis=1)
            max_ratio_corr_with_keep = np.max(ratio_corr_matrix, axis=1)
            max_ratio_corr_featname = [varnames[feat_idx_keep[ratio_argmax_idx]] for ratio_argmax_idx in ratio_argmax_indices]
            # associated max shap correlation and value correlation 
            max_ratio_shapcorr = [shap_corr_matrix_with_kept[i, ratio_argmax_indices[i]] for i in range(len(ratio_argmax_indices))]
            max_ratio_valuecorr = [corr_matrix_with_kept[i, ratio_argmax_indices[i]] for i in range(len(ratio_argmax_indices))]
    

            # get the highest shap value of the 'rand_' random features within shap_meanabs_perfeature
            rand_shap_max = 1e5
            if n_random_vars > 0 and lowershap_than_rdm:
                rand_shap = shap_meanabs_perfeature[[feat_idx_mayexclude[i] for i in range(len(feat_idx_mayexclude)) if 'rand_' in varnames[feat_idx_mayexclude[i]]]]
                rand_shap_max = np.amax(rand_shap)
                print('max shap value of random features', rand_shap_max)
            
            alpha = 0.7
            shap_min = 3e-5
            threshold = 1.2e-4

            # select the features to exclude
            feat_idx_exclude = []
            for i in range(0, len(mayexclude)):
                idx = mayexclude[i]
                shap_this = shap_meanabs_perfeature[feat_idx_mayexclude[idx]]
                if ('rand_' not in varnames[feat_idx_mayexclude[idx]]):
                    print('                  ', f"{idx}: {varnames[feat_idx_mayexclude[idx]]} ------ mean abs SHAP = {shap_this}")
                    print('max corr with previously removed features', np.max(np.abs(corr_badSHAP[feat_idx_exclude, idx]))if len(feat_idx_exclude)>0 else 0)
                    corr_with_kept = max_corr_with_keep[idx]
                    max_shapcorr_this = max_shapcorr_with_keep[idx]
                    shap_mostcorr_feat = shap_meanabs_perfeature[feat_idx_keep[shap_argmax_indices[idx]]]
                    max_ratiocorr_this = max_ratio_corr_with_keep[idx]
                    
                    # if that most correlated feature has already been excluded, then set max_shapcorr_this=0.
                    if len(feat_idx_exclude) > 0:
                        excluded_feature_names = [varnames[feat_idx_mayexclude[excluded_idx]] for excluded_idx in feat_idx_exclude]
                        if max_shapcorr_featname[idx] in excluded_feature_names:
                            max_shapcorr_this = 0.
                            max_ratiocorr_this = 0.

                    print('max corr with kept features', corr_with_kept, '------ feature name:', max_corr_featname[idx])
                    print('max shap value correlation with kept features', max_shapcorr_this, '------ feature name:', max_shapcorr_featname[idx], 
                        '------ shap value of this feature', shap_mostcorr_feat)
                    print('max  shap_corr/(1-valcorr) with kept features', max_ratiocorr_this, '------ feature name:', max_ratio_corr_featname[idx], ', shap corr:', max_ratio_shapcorr[idx], ', value corr:', max_ratio_valuecorr[idx])
                    print('shap / (shap_corr/(1-valcorr))^', f"{alpha:.2f}",' = ', shap_this / pow(max_ratiocorr_this, alpha) if max_ratiocorr_this > 0 else 1e7)
                    corr_rejectedshap = np.max(np.abs(corr_badSHAP[feat_idx_exclude, idx])) if no_corr_with_excluded and len(feat_idx_exclude)>0 else 0
                    addfeat = ((lowshap_highcorr and 
                                ((shap_this-shap_min) / pow(max_ratiocorr_this, alpha) < threshold)
                                #(shap_this < 6e-5 or
                                #(shap_this < 1.5e-4 and max_ratiocorr_this > 0.2) or
                                #(shap_this < 3e-4 and max_ratiocorr_this > 0.5) or
                                #(shap_this < 1.2e-3 and max_ratiocorr_this > 0.8)
                                )
                                    #(shap_this < 6e-5 or 
                                    #    (shap_this < 1.2e-4 and (max_shapcorr_this > 0.25 or corr_with_kept > 0.8)) or 
                                    #    (shap_this < 3e-4 and (max_shapcorr_this > 0.45 or corr_with_kept > 0.92)) or 
                                    #    (shap_this < 1.2e-3 and (max_shapcorr_this > 0.7 or corr_with_kept > 0.98)))
                                or (not lowshap_highcorr)
                                ) and corr_rejectedshap < 0.85 and shap_this < rand_shap_max
                    if addfeat: 
                        feat_idx_exclude.append(idx)
                        print('add', idx)
                    else:   
                        print('skip', idx)
                if len(feat_idx_exclude) == n_toexclude:
                    break

            print(np.array(feat_idx_exclude))
            feat_idx_exclude = feat_idx_mayexclude[np.array(feat_idx_exclude)] if len(feat_idx_exclude) > 0 else []

            '''
            shap_meanabs_perfeature = np.mean(np.abs(shap_values.values), axis=0)
            feature_idx_sortedshap = np.argsort(shap_meanabs_perfeature)
            feat_idx_mayexclude = feature_idx_sortedshap[0:n_shaptolook]
            corr_badSHAP = np.corrcoef(X_test[:, feat_idx_mayexclude], rowvar=False)
            feat_idx_exclude = [0]
            for i in range(1, len(feat_idx_mayexclude)):
                print('corrs with previous vars', corr_badSHAP[feat_idx_exclude, i])
                if np.max(np.abs(corr_badSHAP[feat_idx_exclude, i])) < 0.6:
                    feat_idx_exclude.append(i)
                    print('add',i)
                if len(feat_idx_exclude) == n_toexclude:
                    break
            print(np.array(feat_idx_exclude))
            feat_idx_exclude = feat_idx_mayexclude[np.array(feat_idx_exclude)]
            '''
            
            print(np.array(feat_idx_exclude))
            print('lowest abs mean shap value = ', shap_meanabs_perfeature[feat_idx_exclude])

            feature_names_toexclude = varnames[feat_idx_exclude]
            print('write down these variables for future exclusion: ', feature_names_toexclude)
            with open(os.path.join(var.DATA_PATH, 'excluded_variables_fromSHAP.txt'), 'a') as f:
                for line in feature_names_toexclude:
                    f.write(f"{line}\n")

        print('shap_values')
        f = plt.figure(num=1, clear=True)
        targetsort = np.argsort(y_test)[::-1]
        targetsort_onlyStableTimes = np.argsort(y_test[inds_onlyStableTimes])[::-1]

        print('shap figure 1')
        plt.figure(num=1, clear=True)
        shap.plots.bar(shap_eachvar, show=False, max_display=45)
        plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_bar_timeIntegrated'+('_excludeworstSHAP' if exclude_timefeats else '')+'.pdf'), dpi=250, bbox_inches='tight')

        #plt.figure(num=1, clear=True)
        #shap.plots.beeswarm(shap_eachrange, show=False, max_display=40)
        #plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_beeswarm_perTimerange.pdf'), dpi=250, bbox_inches='tight')

        print('shap figure 2')
        plt.figure(num=1, clear=True)
        shap.plots.bar(shap_eachrange, show=False, max_display=45)
        plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_bar_perTimerange'+('_excludeworstSHAP' if exclude_timefeats else '')+'.pdf'), dpi=250, bbox_inches='tight')

        #plt.figure(num=1, clear=True)
        #shap.plots.beeswarm(shap_eachrange_coarse, show=False, max_display=40)
        #plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_beeswarm_perCoarseTimerange.pdf'), dpi=250, bbox_inches='tight')

        print('shap figure 3')
        plt.figure(num=1, clear=True)
        shap.plots.bar(shap_eachrange_coarse, show=False, max_display=45)
        plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_bar_perCoarseTimerange'+('_excludeworstSHAP' if exclude_timefeats else '')+'.pdf'), dpi=250, bbox_inches='tight')
            
        print('shap figure 4')
        plt.figure(num=1, clear=True)
        shap.plots.bar(shap_values, show=False, max_display=45) 
        plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_bar'+('_excludeworstSHAP' if exclude_timefeats else '')+'.pdf'), dpi=250, bbox_inches='tight')

        print('shap figure 5')
        plt.figure(num=1, clear=True)
        shap.plots.bar(shap_values, show=False, max_display=350)
        plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_bar_allvars'+('_excludeworstSHAP' if exclude_timefeats else '')+'.pdf'), dpi=250, bbox_inches='tight')

        #print('shap figure 6')
        #plt.figure(num=1, clear=True)
        #shap.plots.beeswarm(shap_values, show=False, max_display=40)
        #plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_beeswarm'+('_excludeworstSHAP' if exclude_timefeats else '')+'.pdf'), dpi=250, bbox_inches='tight')

        #if not exclude_timefeats:
        #    print('shap figure 7')
        #    plt.figure(num=1, clear=True)
        #    shap.plots.beeswarm(shap_eachvar, show=False, max_display=40)
        #    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_beeswarm_timeIntegrated.pdf'), dpi=250, bbox_inches='tight')

        #plt.figure(num=1, clear=True)
        #shap.plots.beeswarm(shap_values, show=False, max_display=228)
        #plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_beeswarm_allvars.pdf'), dpi=250, bbox_inches='tight')
        #plt.figure(num=1, clear=True)

        plot_SHAP_eachfeature = False
        if plot_SHAP_eachfeature:
            print('Scatter plot of all instances for each feature')
            for v in range(shap_values.shape[1]):
                print(v, varnames[v])
                #SHAP_featureDependence(y_test, shap_values, targetsort, varnames[v], v, ml_param, 3600, onlyStableTimes=False)
                SHAP_featureDependence(y_test[inds_onlyStableTimes], shap_values[inds_onlyStableTimes], targetsort_onlyStableTimes, varnames[v], v, ml_param, 3600, onlyStableTimes=True)
            print('shap_eachvar')
            for v in range(shap_eachvar.shape[1]):
                print(v, shap_eachvar.feature_names[v])
                #SHAP_featureDependence(y_test, shap_eachvar, targetsort, varnames_eachvar[v], v, ml_param, 3600, onlyStableTimes=False)
                SHAP_featureDependence(y_test[inds_onlyStableTimes], shap_eachvar[inds_onlyStableTimes], targetsort_onlyStableTimes, shap_eachvar.feature_names[v], v, ml_param, 3600, onlyStableTimes=True)


        #shap.plots.bar(shap_values[y_test[passClassif_test] < ml_param.transform_target(6*3600)], show=False, max_display=142)
        #plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_bar_trueEarlinessBelow6h.pdf'), dpi=250, bbox_inches='tight')
        #plt.figure()
        #shap.plots.bar(shap_values[y_test[passClassif_test] >= ml_param.transform_target(6*3600)], show=False, max_display=142)
        #plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_bar_trueEarlinessAbove6h.pdf'), dpi=250, bbox_inches='tight')
        #plt.figure(num=1, clear=True)
        #shap.plots.beeswarm(shap_values[y_test < ml_param.transform_target(6*3600)], show=False, max_display=142)
        #plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_beeswarm_trueEarlinessBelow6h.pdf'), dpi=250, bbox_inches='tight')
        #plt.figure(num=1, clear=True)
        #shap.plots.beeswarm(shap_values[y_test >= ml_param.transform_target(6*3600)], show=False, max_display=142)
        #plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP','SHAP_beeswarm_trueEarlinessAbove6h.pdf'), dpi=250, bbox_inches='tight')
        

        '''
        # check feature importance
        plot_features_importance(model, varnames, 'total_gain', log_imp=True)
        plot_features_importance(model, varnames, 'total_cover', log_imp=True)
        plot_features_importance(model, varnames, 'weight')
        plot_features_importance(model, varnames, 'gain', log_imp=True)
        plot_features_importance(model, varnames, 'cover', log_imp=True)
        plot_features_importance(model, varnames, 'total_gain', imp_sort=True, log_imp=True)
        plot_features_importance(model, varnames, 'total_cover', imp_sort=True, log_imp=True)
        plot_features_importance(model, varnames, 'weight', imp_sort=True)
        plot_features_importance(model, varnames, 'gain', imp_sort=True, log_imp=True)
        plot_features_importance(model, varnames, 'cover', imp_sort=True, log_imp=True)
        '''

