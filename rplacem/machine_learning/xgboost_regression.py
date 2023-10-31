import numpy as np
import xgboost as xg
import os
import math
import pickle
import matplotlib.colors as colors
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import rplacem.globalvariables_peryear as vars
var = vars.var
import pandas as pd
import json
import pprint
from matplotlib.patches import Rectangle
from scipy.integrate import simpson
import shap
import copy

# PLOTTING ROUTINES

def scatter_true_vs_pred(true, pred, id_idx, test_inds, loge=True, onlysomecompos=False, trainsamples=False, addsavename=''):
    '''
    2D plot of earliness: true vs predicted
    '''
    fig = plt.figure()
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
                      trainsamples=False, normalize_cols=True, addsavename=''):

    fig = plt.figure()
    sns.despine()

    plt.ylabel(r'100 + $\bf{predicted}$ earliness [s]', fontsize=16)
    plt.xlabel(r'100 + $\bf{true}$ earliness [s]', fontsize=16)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    emin = 100.
    emax = 2.8e5
    plt.xlim(emin, emax)
    plt.ylim(emin, emax)
    if loge:
        plt.xscale('log')
        plt.yscale('log')


    H, xedges, yedges = np.histogram2d( 100. + true,
                                        100. + pred,
                                        bins=[np.logspace(np.log10(emin), np.log10(emax), 50),
                                        np.logspace(np.log10(emin), np.log10(emax), 50)])
    H = H.T
    # normalize by columns
    if normalize_cols:
        H = H / np.sum(H, axis=0)
    X, Y = np.meshgrid(xedges, yedges)
    vm = max(1e-3 if normalize_cols else 1., np.nanmin(H[H > 0]))
    vM = min(np.nanmax(H), zmax)
    H[(H < 1e-3) & (H > 3e-4)] = 1e-3
    H[H < 3e-4] = np.NaN

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

    xminrect = 1.1*6*3600
    plt.gca().add_patch(Rectangle((xminrect, 100), emax-xminrect, emax-100, alpha=0.75, color='white', linewidth=0))
    plt.text(7e4, 1000, r'$\bf{not\ \ predictable}$', color='green', fontsize=14, rotation=90)
    
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

def correlation_heatmap(correlations, summary=False, zoom_extremes=True):
    fig= plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, vmin=-1.0, center=0, cmap=extremes_colormap(60 if zoom_extremes else 0), fmt='.2f', 
                square=True, linewidths=.5, annot=False, cbar_kws={"shrink": .70},
                xticklabels=varnames,
                yticklabels=varnames,
                )
    plt.xticks(fontsize=4.5)
    plt.yticks(fontsize=4.5)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'correlation_matrix'+('_timevaraverage' if summary else '')+'.pdf'), bbox_inches='tight')

def plot_probas_BinaryClassification(sig, bkg, e_thres_str):
    plt.figure()
    hsig = plt.hist(sig, bins=np.linspace(0,1,70), histtype = 'step', label='true earliness < '+e_thres_str, density=True)
    hbkg = plt.hist(bkg, bins=np.linspace(0,1,70), histtype = 'step', label='true earliness > '+e_thres_str, density=True)
    plt.legend()
    plt.xlabel('xbgoost output')
    plt.ylabel('p.d.f.')
    plt.xlim([0,1])
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'proba_SigBkg_BinaryClassif_earlinessBelow'+e_thres_str+'.png'), dpi=250)
    return hsig, hbkg

def plotROC_binaryClassification(sig, bkg, e_thres_str):
    npts = 250
    p_probe = np.linspace(0,1,npts+1)
    TPR = np.ones(npts+2)
    FPR = np.ones(npts+2)
    TPR[-1] = 0; FPR[-1] = 0
    nsig = len(sig)
    nbkg = len(bkg)
    for i, p in enumerate(p_probe):
        TPR[1+i] = np.count_nonzero(sig > p) / nsig
        FPR[1+i] = np.count_nonzero(bkg > p) / nbkg
    
    plt.figure(figsize=(6,6))
    plt.plot(FPR, TPR)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('false positive rate', fontsize=16)
    plt.ylabel('true positive rate', fontsize=16)
    plt.title('ROC', fontsize=16)
    plt.plot(np.linspace(0,1,2), np.linspace(0,1,2), color='black', linestyle=':', linewidth=2)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'ROC_SigBkg_BinaryClassif_earlinessBelow'+e_thres_str+'.png'), dpi=250, bbox_inches='tight')

    return TPR, FPR

    








# transformation of researched output
logsub_transform = 1.5
def transform_target(targ):
    return np.log10(100. + targ) - logsub_transform

def invtransform_target(targ):
    return np.power(10, targ + logsub_transform) - 100.







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

def negF1(pred, dtrain, e_thres):
    true = dtrain.get_label()
    TPR = np.count_nonzero((true < e_thres) & (pred < e_thres)) / np.count_nonzero(true < e_thres)
    purity = np.count_nonzero((true < e_thres) & (pred < e_thres)) / np.count_nonzero(pred < e_thres)
    F1 = 2*purity*TPR / (purity+TPR)
    return float(-F1)

def negF1_20min(pred, dtrain):
    e_thres = transform_target(1200)
    return 'F1_20min', negF1(pred, dtrain, e_thres)

def negF1_1h(pred, dtrain):
    e_thres = transform_target(3600)
    return 'F1_1h', negF1(pred, dtrain, e_thres)

def negF1_3h(pred, dtrain):
    e_thres = transform_target(3*3600)
    return 'F1_3h', negF1(pred, dtrain, e_thres)

def ROCAUC(pred, dtrain, e_thres, maxFPR=1.):
    '''
    Calculate the true and false positive rates for various thresholds on the predicted earliness, 
    for a given e_thres on the true earliness (that gives the binary classification).
    Then calculate the area under the curve for this ROC.
    Returns the negative ROC normalized by the max FPR
    For perfect discrimination, this equals -1, and for the baseline (no discrimination), it equals -maxFPR/2
    '''
    true = dtrain.get_label()

    thresholds = transform_target(
                 np.array([400, 600, 900, 1200, 1600, 2000, 2400, 3000, 3600, 
                           1.3*3600, 1.6*3600, 2*3600, 3*3600, 4*3600, 5*3600, 6*3600, 7*3600, 8*3600, 9*3600, 10*3600, 12*3600, 15*3600]))
    n_thres = thresholds.size
    TPR = np.zeros(n_thres+2)
    FPR = np.zeros(n_thres+2)
    TPR[0] = 0; TPR[-1] = 1
    FPR[0] = 0; FPR[-1] = 1
    is_sig = (true < e_thres)
    is_bkg = (true >= e_thres)

    for i in range(n_thres):
        detected = (pred < thresholds[i])
        TPR[1+i] = np.count_nonzero(is_sig & detected) / np.count_nonzero(is_sig)
        FPR[1+i] = np.count_nonzero(is_bkg & detected) / np.count_nonzero(is_bkg)

    if maxFPR < 1: # compute area only under value maxFPR 
        TPR_atMaxFPR = np.interp(maxFPR, FPR, TPR)
        TPR = np.append(TPR[FPR <= maxFPR], TPR_atMaxFPR)
        FPR = np.append(FPR[FPR <= maxFPR], maxFPR)

    # simpson is integrating under a 2nd-order polynomial interpolation of data
    ROC = simpson(TPR, TPR)
    return - ROC / maxFPR

def ROCAUC_1h_maxFPR0p3(pred, dtrain):
    e_thres = transform_target(3600)
    maxFPR = 0.3
    return 'ROC_1h', ROCAUC(pred, dtrain, e_thres, maxFPR)

def ROCAUC_20min_maxFPR0p3(pred, dtrain):
    e_thres = transform_target(1200)
    maxFPR = 0.3
    return 'ROC_20min', ROCAUC(pred, dtrain, e_thres, maxFPR)

def ROCAUC_3h_maxFPR0p3(pred, dtrain):
    e_thres = transform_target(3*3600)
    maxFPR = 0.3
    return 'ROC_3h', ROCAUC(pred, dtrain, e_thres, maxFPR)

def ROCAUC_3h_maxFPR0p4(pred, dtrain):
    e_thres = transform_target(3*3600)
    maxFPR = 0.4
    return 'ROC_3h', ROCAUC(pred, dtrain, e_thres, maxFPR)

def ROCAUC_6h_maxFPR1(pred, dtrain):
    e_thres = transform_target(3*3600)
    maxFPR = 6
    return 'ROC_6h', ROCAUC(pred, dtrain, e_thres, maxFPR)







# extract training data from file
file_path = os.path.join(var.DATA_PATH, 'training_data_142variables.pickle')
with open(file_path, 'rb') as f:
    [inputvals, outputval, varnames, eventtime, id_idx, id_dict
     #, coarse_timerange, n_traintimes, n_traintimes_coarse
     ] = pickle.load(f) 
coarse_timerange = np.array([0,0,0,0,0,0,1,1,0,0,0,1,1,0], dtype=bool)
n_traintimes = 11
n_traintimes_coarse = 8

#previous_tstep_idx

# temporary, until re-running build_datasets
min_earliness_notrans = 12*3600 # minimum random earliness given to events in no-transition compositions
max_earliness = var.TIME_WHITEOUT - 6*3600 # max range of earliness
notrans_events = np.where(np.isclose(outputval, 2e5, atol=1e-4))[0]
outputval[notrans_events] = np.random.uniform(min_earliness_notrans, max_earliness, len(notrans_events))


# implement a small max_earliness
#maxearliness = 12*3600
#outputval[outputval > maxearliness] = maxearliness

# initial datasets
nmaxcomp = 1e5

select = np.where(np.array(id_idx) < nmaxcomp)
outputval_orig = np.array(outputval[select])
outputval = transform_target(outputval_orig)
inputvals = inputvals[select]
id_idx = np.array(id_idx)[select]
ncomp = np.max(id_idx)+1
n_events = len(outputval)
print('use',n_events,'events from',ncomp,'compositions for training')
nvar = len(varnames)
print('use',nvar,'features')

# build event weights
weights = np.ones(n_events)
wmax = 1
wmin = 0.25
xbreak1 = 2e4
xbreak2 = min_earliness_notrans
weights[outputval_orig <= xbreak1] = wmax
weights[outputval_orig >= xbreak2] = wmin
# power-law interpolation between these two limits
powerexp = math.log(wmax / wmin) / math.log(xbreak1 / xbreak2)
powerlaw_inds = np.where((outputval_orig > xbreak1) & (outputval_orig < xbreak2))
weights[powerlaw_inds] = wmax * np.power(outputval_orig[powerlaw_inds] / xbreak1, powerexp)
sumweights = np.sum(weights)

# event weights with higher wmin
wmin_high = 0.4
weights_highwmin = np.copy(weights)
weights[outputval_orig >= xbreak2] = wmin_high
powerexp = math.log(wmax / wmin_high) / math.log(xbreak1 / xbreak2)
weights_highwmin[powerlaw_inds] = wmax * np.power(outputval_orig[powerlaw_inds] / xbreak1, powerexp)
sumweights_highwmin = np.sum(weights_highwmin)


# Split the data into training and testing sets
test_size = 0.2 # 20% of data for testing
# need to split in terms of compositions and not events, so that there are no train-test correlations (ie events from the same compo both in train and test sample)
rdm = np.random.rand(n_events) # random between 0 and 1
np.random.seed(42)
shuf = np.random.permutation(ncomp)
test_inds = np.where(shuf[id_idx] < test_size * ncomp)[0]
train_inds = np.where(shuf[id_idx] >= test_size * ncomp)[0]
print('#events in test sample', len(test_inds))
print('#events in train sample', len(train_inds))

# create split train/test samples
X_train = inputvals[train_inds]
X_test = inputvals[test_inds]
y_train = outputval[train_inds]
y_test = outputval[test_inds]
w_train = weights[train_inds]
w_test = weights[test_inds]
w_highwmin_train = weights_highwmin[train_inds]
w_highwmin_test = weights_highwmin[test_inds]




# INITIAL BINARY CLASSIFICATION
binaryClassif = False

if binaryClassif:
    # create samples for binary classification, without weights
    classif_e_threshold_h = 6
    classif_e_threshold = transform_target(classif_e_threshold_h*3600.) # 6 hours maximal earliness for true warnings
    y_train_classif = (outputval[train_inds] < classif_e_threshold).astype(int)
    y_test_classif = (outputval[test_inds] < classif_e_threshold).astype(int)
    dtrain_classif = xg.DMatrix(data = X_train, label = y_train_classif, feature_names=varnames)
    dtest_classif  = xg.DMatrix(data = X_test,  label = y_test_classif,  feature_names=varnames)

    # Set XGBoost regressor parameters
    num_rounds = 140 if (classif_e_threshold_h == 6) else 170 # Max number of boosting rounds (iterations)
    maxdepth = 7
    learningrate = 0.11
    subsample = 0.8
    colsample = 0.75
    minchildw = 4
    params = {
        'objective': 'binary:logistic',            # Objective function
        'max_depth': maxdepth,                    # Maximum depth of each tree
        'learning_rate': learningrate,              # Learning rate
        'subsample': subsample,                  # Subsample ratio of the training instances
        'colsample_bytree': colsample,           # Subsample ratio of features at each new tree
        'colsample_bylevel': colsample,          # Subsample ratio of features at each tree level
        'min_child_weight': minchildw,             # minimum sum of event weights (times the hessian value) per node
        'random_state': 42,                # Random seed
        'eval_metric': ['auc', 'logloss'], 
        #'verbosity' : 2
    }

    # Train the model
    eval_res_classif = {}
    model_classif = xg.train(params, dtrain_classif, 
                    num_boost_round=num_rounds,
                    early_stopping_rounds=8,
                    evals=[(dtest_classif, 'test'), (dtrain_classif, 'train')],
                    evals_result=eval_res_classif
                    )

    # Evaluate the model on the testing set
    predictions_classif = model_classif.predict(dtest_classif)
    predictions_train_classif = model_classif.predict(dtrain_classif)

    e_thres_str = ( str(int(classif_e_threshold_h))+'h' if classif_e_threshold_h > 0.9 else str(int(classif_e_threshold_h*60+0.2))+'min')
    hsig, hbkg = plot_probas_BinaryClassification(predictions_classif[y_test_classif > 0.5], predictions_classif[y_test_classif < 0.5], e_thres_str)
    TPR, FPR = plotROC_binaryClassification(predictions_classif[y_test_classif > 0.5], predictions_classif[y_test_classif < 0.5], e_thres_str)
    print('final auc train/test = ', eval_res_classif['train']['auc'][-1] / eval_res_classif['test']['auc'][-1])
    cumhsig = np.cumsum(hsig[0]) / np.sum(hsig[0])
    cumhbkg = np.cumsum(hbkg[0]) / np.sum(hbkg[0])

    # model output that separates signal in halves
    minprobsig = 0.5 # minimum classification output for being in the signal-enriched sample
    halfSig_outputval = np.interp(1-minprobsig, cumhsig, hsig[1][:-1] + np.diff(hsig[1])/2) # compare to bin centers
    halfSig_FPR = 1 - np.interp(1-minprobsig, cumhsig, cumhbkg) # model output that separates signal in halves
    print('fraction',minprobsig,'of signal is kept for output >', halfSig_outputval,'with FPR',halfSig_FPR)

    plt.figure()
    xlog=False
    bins = np.logspace(np.log10(1e2), np.log10(2.8e5), 200) if xlog else np.linspace(0, 2.8e5, 200)
    plt.hist(invtransform_target( y_train[predictions_train_classif < halfSig_outputval] ), bins=bins,  histtype = 'step')
    plt.hist(invtransform_target( y_train[predictions_train_classif >= halfSig_outputval] ), bins=bins,  histtype = 'step')
    if xlog:
        plt.xscale('log')
    plt.xlim([1e2 if xlog else 0, 2.8e5])
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'earliness_trainsample_VS_classificationOutput.png'), dpi=250, bbox_inches='tight')

    plt.figure()
    plt.hist(invtransform_target( y_train[predictions_train_classif < halfSig_outputval] ), bins=np.linspace(0, 2.8e5, 200),  histtype = 'step')
    plt.xlim([0, 2.8e5])
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'earliness_trainsample.png'), dpi=250, bbox_inches='tight')

    # save TPR and FPR for later
    with open(os.path.join(var.DATA_PATH, 'MLClassificationEvaluation'
                                        +'_earlinessBelow'+e_thres_str
                                        +'_maxdepth'+str(maxdepth)
                                        +'_minchildweight'+str(minchildw)
                                        +'_subsample'+str(subsample)
                                        +'_colsample'+str(colsample)
                                        +'_logsubtransform'+str(logsub_transform)
                                        +'.json'), 'w') as f:

        json.dump({'TPR':list(TPR), 'FPR':list(FPR)}, f)







noregression = False
if noregression:
    exit()

# REGRESSION FOR CONTINUOUS EARLINESS PREDICTION

# indices of events that pass or not the condition for being in the signal-enriched sample
passClassif_test = np.where(predictions_classif >= halfSig_outputval)[0] if binaryClassif else np.arange(0,len(X_test))
failClassif_test = np.where(predictions_classif < halfSig_outputval)[0] if binaryClassif else np.array([])
passClassif_train = np.where(predictions_train_classif >= halfSig_outputval)[0] if binaryClassif else np.arange(0,len(X_train))
failClassif_train = np.where(predictions_train_classif < halfSig_outputval)[0] if binaryClassif else np.array([])

# Datasets for regression on signal-enriched data
dtrain_regSig = xg.DMatrix(data = X_train[passClassif_train], label = y_train[passClassif_train], weight=w_highwmin_train[passClassif_train], feature_names=varnames)
#dtrain_regSig.set_uint_info('previous_event_idx', previous_tstep_idx[passClassif_train])
dtest_regSig = xg.DMatrix(data = X_test[passClassif_test], label = y_test[passClassif_test], weight=w_highwmin_test[passClassif_test], feature_names=varnames)
if binaryClassif:
    # Datasets for regression on background-enriched data
    dtrain_regBkg = xg.DMatrix(data = X_train[failClassif_train], label = y_train[failClassif_train], weight=w_train[failClassif_train], feature_names=varnames)
    #dtrain_regBkg.set_uint_info('previous_event_idx', previous_tstep_idx[failClassif_train])
    dtest_regBkg = xg.DMatrix(data = X_test[failClassif_test], label = y_test[failClassif_test], weight=w_test[failClassif_test], feature_names=varnames)

# Set XGBoost regressor parameters (same for the 2 models)
num_rounds = 90  # Max number of boosting rounds (iterations)
maxdepth = 8
learningrate = 0.1
subsample = 0.8
colsample = 0.75
minchildw = 3.1 # careful, this should be tuned again if wmin is changed
params = {
    #'objective': 'reg:squarederror',  # Regression task with squared loss # or: squarederror, squaredlogerror, absoluteerror # this should be 'objective' rather than 'obj' for sklearn
    'max_depth': maxdepth,                    # Maximum depth of each tree
    'learning_rate': learningrate,              # Learning rate
    'subsample': subsample,                  # Subsample ratio of the training instances
    'colsample_bytree': colsample,           # Subsample ratio of features at each new tree
    'colsample_bylevel': colsample,          # Subsample ratio of features at each tree level
    'min_child_weight': minchildw,             # minimum sum of event weights (times the hessian value) per node
    'random_state': 42,                # Random seed
    #'eval_metric': ['mae','mphe','rmsle','rmse','mape']#meanrelerr, rmsrelerr, 
    #'verbosity' : 2
}

# Train the 2 models
print('start training regression for signal enriched data, with',len(passClassif_train),'events')
eval_res_Sig = {}
model_Sig = xg.train(params, dtrain_regSig, 
                     num_boost_round=num_rounds,
                     early_stopping_rounds=25,
                     obj=squaredrelerr,
                     custom_metric=ROCAUC_3h_maxFPR0p3,
                     evals=[(dtest_regSig, 'test'), (dtrain_regSig, 'train')],
                     evals_result=eval_res_Sig
                    )
print('done training')  

if binaryClassif:
    print('start training regression for background enriched data with',len(failClassif_train),'events')
    eval_res_Bkg = {}
    params['learning_rate'] = 0.06
    params['min_child_weight'] = 1.1
    model_Bkg = xg.train(params, dtrain_regBkg, 
                        num_boost_round=num_rounds,
                        early_stopping_rounds=20,
                        obj=squaredrelerr,
                        custom_metric=ROCAUC_6h_maxFPR1,
                        evals=[(dtest_regBkg, 'test'), (dtrain_regBkg, 'train')],
                        evals_result=eval_res_Bkg
                        )
    print('done training')








# EVALUATION
# weights are in general ignored here

# Evaluate the models on the testing sets
predictions_regSig = model_Sig.predict(dtest_regSig)
predictions_regSig_train = model_Sig.predict(dtrain_regSig)
if binaryClassif:
    predictions_regBkg = model_Bkg.predict(dtest_regBkg)
    predictions_regBkg_train = model_Bkg.predict(dtrain_regBkg)


# functions for recording and printing evaluations
def evals(true, pred, e_thres, e_true_min=0):
    if e_thres <= e_true_min:
        e_true_min = 0
    TPR = []
    FPR = []
    purity = []
    F1 = []
    thres_list = [e_thres, 400, 600, 900, 1200, 1800, 2400, 3000, 3600, 1.3*3600, 1.6*3600, 2*3600, 3*3600, 4*3600, 5*3600, 6*3600, 7*3600, 8*3600, 10*3600]
    for i, pred_thres in enumerate(thres_list):
        truepos = np.count_nonzero((true < e_thres) & (true > e_true_min))
        TPR.append(np.count_nonzero((true < e_thres) & (true > e_true_min) & (pred < pred_thres)) / truepos if truepos > 0 else -1)
        falsepos = np.count_nonzero(true >= e_thres)
        FPR.append(np.count_nonzero((true >= e_thres) & (pred < pred_thres)) / falsepos if falsepos > 0 else -1)
        predpos = np.count_nonzero(pred < pred_thres)
        purity.append(np.count_nonzero((true < e_thres) & (pred < pred_thres)) / predpos if predpos > 0 else -1)
        F1.append(2*purity[i]*TPR[i] / (purity[i]+TPR[i]) if (purity[i] > 0 and TPR[i] > 0) else -1)
    frequency = np.count_nonzero(true < e_thres) / np.count_nonzero(true)
    return {'TPR':TPR, 'FPR':FPR, 'purity':purity, 'F1':F1, 'threshold_list':thres_list, 'e_true_min':e_true_min, 'frequency':frequency}

def print_mainevals(eval, printFreq=True):
    print('             TPR =','%.4f' % eval['TPR'][0], ', FPR =','%.4f' % eval['FPR'][0], ', purity =','%.4f' % eval['purity'][0], ', F1 = ','%.4f' % eval['F1'][0], (', frequency = '+'%.4f' % eval['frequency'] if printFreq else ''))

def divide_evalres(eval1, eval2):
    newTPR = list(np.array(eval1['TPR']) / np.array(eval2['TPR']))
    newFPR = list(np.array(eval1['FPR']) / np.array(eval2['FPR']))
    newpurity = list(np.array(eval1['purity']) / np.array(eval2['purity']))
    newF1 = list(np.array(eval1['F1']) / np.array(eval2['F1']))
    newfrequency = eval1['frequency'] / eval2['frequency']
    return {'TPR':newTPR, 'FPR':newFPR, 'purity':newpurity, 'F1':newF1, 'threshold_list':eval1['threshold_list'], 'frequency':newfrequency}

# return earliness from its transformation used in training
true_test_regSig = invtransform_target(y_test[passClassif_test])
pred_test_regSig = invtransform_target(predictions_regSig)
true_train_regSig = invtransform_target(y_train[passClassif_train])
pred_train_regSig = invtransform_target(predictions_regSig_train)
if binaryClassif:
    true_test_regBkg = invtransform_target(y_test[failClassif_test])
    pred_test_regBkg = invtransform_target(predictions_regBkg)
    true_train_regBkg = invtransform_target(y_train[failClassif_train])
    pred_train_regBkg = invtransform_target(predictions_regBkg_train)

    hist_true_vs_pred(true_train_regBkg, pred_train_regBkg, zmax=1.3e3, 
                    loge=True, normalize_cols=False, addsavename='_backgroundEnriched_train')
    hist_true_vs_pred(true_test_regBkg, pred_test_regBkg, zmax=1.3e3, 
                    loge=True, normalize_cols=False, addsavename='_backgroundEnriched')



# write evaluation results in a json
with open(os.path.join(var.DATA_PATH, 'MLevaluation'
                                     +'_weightSigEnriched-high-e-0p'+str(int(wmin_high*100))
                                     +'_weightBkgEnriched-high-e-0p'+str(int(wmin*100))
                                     +'_maxdepth'+str(maxdepth)
                                     +'_minchildweight'+str(minchildw)
                                     +'_subsample'+str(subsample)
                                     +'_colsample'+str(colsample)
                                     +'_logsubtransform'+str(logsub_transform)
                                     +'.json'), 'w') as f:
    eval_out = {'20min_regSig': evals(true_test_regSig, pred_test_regSig, 1200),
               '1h_regSig': evals(true_test_regSig, pred_test_regSig, 3600),
               '3h_regSig': evals(true_test_regSig, pred_test_regSig, 3*3600),
               '6h_regSig': evals(true_test_regSig, pred_test_regSig, 6*3600),
               '1-3h_regSig': evals(true_test_regSig, pred_test_regSig, 3*3600, 3600),
               '3-6h_regSig': evals(true_test_regSig, pred_test_regSig, 6*3600, 3*3600),
               '20min_train_regSig': evals(true_train_regSig, pred_train_regSig, 1200),
               '1h_train_regSig': evals(true_train_regSig, pred_train_regSig, 3600),
               '3h_train_regSig': evals(true_train_regSig, pred_train_regSig, 3*3600),
               '6h_train_regSig': evals(true_train_regSig, pred_train_regSig, 6*3600),
               '1-3h_train_regSig': evals(true_train_regSig, pred_train_regSig, 3*3600, 3600),
               '3-6h_train_regSig': evals(true_train_regSig, pred_train_regSig, 6*3600, 3*3600),
               'stdeval_regSig': dict(zip(eval_res_Sig['test'].keys(), list(eval_res_Sig['test'].values())[-1])),
               #'20min_regBkg': evals(true_test_regBkg, pred_test_regBkg, 1200),
               #'1h_regBkg': evals(true_test_regBkg, pred_test_regBkg, 3600),
               #'3h_regBkg': evals(true_test_regBkg, pred_test_regBkg, 3*3600),
               #'6h_regBkg': evals(true_test_regBkg, pred_test_regBkg, 6*3600),
               #'1-3h_regBkg': evals(true_test_regBkg, pred_test_regBkg, 3*3600, 3600),
               #'3-6h_regBkg': evals(true_test_regBkg, pred_test_regBkg, 6*3600, 3*3600),
               #'20min_train_regBkg': evals(true_train_regBkg, pred_train_regBkg, 1200),
               #'1h_train_regBkg': evals(true_train_regBkg, pred_train_regBkg, 3600),
               #'3h_train_regBkg': evals(true_train_regBkg, pred_train_regBkg, 3*3600),
               #'6h_train_regBkg': evals(true_train_regBkg, pred_train_regBkg, 6*3600),
               #'1-3h_train_regBkg': evals(true_train_regBkg, pred_train_regBkg, 3*3600, 3600),
               #'3-6h_train_regBkg': evals(true_train_regBkg, pred_train_regBkg, 6*3600, 3*3600),
               #'stdeval_regBkg': dict(zip(eval_res_Bkg['test'].keys(), list(eval_res_Bkg['test'].values())[-1]))
               }
    
    eval_out['20min_trainVsTest_regSig'] = divide_evalres(eval_out['20min_train_regSig'], eval_out['20min_regSig'])
    eval_out['1h_trainVsTest_regSig'] = divide_evalres(eval_out['1h_train_regSig'], eval_out['1h_regSig'])
    eval_out['3h_trainVsTest_regSig'] = divide_evalres(eval_out['3h_train_regSig'], eval_out['3h_regSig'])
    eval_out['6h_trainVsTest_regSig'] = divide_evalres(eval_out['6h_train_regSig'], eval_out['6h_regSig'])
    eval_out['1-3h_trainVsTest_regSig'] = divide_evalres(eval_out['1-3h_train_regSig'], eval_out['1-3h_regSig'])
    eval_out['3-6h_trainVsTest_regSig'] = divide_evalres(eval_out['3-6h_train_regSig'], eval_out['3-6h_regSig'])
    
    #eval_out['20min_trainVsTest_regBkg'] = divide_evalres(eval_out['20min_train_regBkg'], eval_out['20min_regBkg'])
    #eval_out['1h_trainVsTest_regBkg'] = divide_evalres(eval_out['1h_train_regBkg'], eval_out['1h_regBkg'])
    #eval_out['3h_trainVsTest_regBkg'] = divide_evalres(eval_out['3h_train_regBkg'], eval_out['3h_regBkg'])
    #eval_out['6h_trainVsTest_regBkg'] = divide_evalres(eval_out['6h_train_regBkg'], eval_out['6h_regBkg'])
    #eval_out['1-3h_trainVsTest_regBkg'] = divide_evalres(eval_out['1-3h_train_regBkg'], eval_out['1-3h_regBkg'])
    #eval_out['3-6h_trainVsTest_regBkg'] = divide_evalres(eval_out['3-6h_train_regBkg'], eval_out['3-6h_regBkg'])
    json.dump(eval_out, f)

    #pprint.pprint(eval_out)
    #print('evaluation for signal-enriched sample')
    print('earliness < 20min, test:')
    print_mainevals(eval_out['20min_regSig'])
    print('earliness < 20min, train/test:')
    print_mainevals(eval_out['20min_trainVsTest_regSig'], False)
    print('earliness < 1h, test:')
    print_mainevals(eval_out['1h_regSig'])
    print('earliness < 1h, train/test:')
    print_mainevals(eval_out['1h_trainVsTest_regSig'], False)
    print('earliness < 3h, test:')
    print_mainevals(eval_out['3h_regSig'])
    print('earliness < 3h, train/test:')
    print_mainevals(eval_out['3h_trainVsTest_regSig'], False)
    print('earliness < 6h, test:')
    print_mainevals(eval_out['6h_regSig'])
    print('earliness < 6h, train/test:')
    print_mainevals(eval_out['6h_trainVsTest_regSig'], False)
    print('1h < earliness < 3h, test:')
    print_mainevals(eval_out['1-3h_regSig'])
    print('1h < earliness < 3h, train/test:')
    print_mainevals(eval_out['1-3h_trainVsTest_regSig'], False)
    print('3h < earliness < 6h, test:')
    print_mainevals(eval_out['3-6h_regSig'])
    print('3h < earliness < 6h, train/test:')
    print_mainevals(eval_out['3-6h_trainVsTest_regSig'], False)
    pprint.pprint(eval_out['stdeval_regSig'])

    #print('evaluation for background-enriched sample')
    #print('earliness < 20min, test:')
    #print_mainevals(eval_out['20min_regBkg'])
    #rint('earliness < 20min, train/test:')
    #print_mainevals(eval_out['20min_trainVsTest_regBkg'], False)
    #print('earliness < 1h, test:')
    #print_mainevals(eval_out['1h_regBkg'])
    #print('earliness < 1h, train/test:')
    #print_mainevals(eval_out['1h_trainVsTest_regBkg'], False)
    #print('earliness < 3h, test:')
    #print_mainevals(eval_out['3h_regBkg'])
    #print('earliness < 3h, train/test:')
    #print_mainevals(eval_out['3h_trainVsTest_regBkg'], False)
    #print('earliness < 6h, test:')
    #print_mainevals(eval_out['6h_regBkg'])
    #print('earliness < 6h, train/test:')
    #print_mainevals(eval_out['6h_trainVsTest_regBkg'], False)
    #print('1h < earliness < 3h, test:')
    #print_mainevals(eval_out['1-3h_regBkg'])
    #print('1h < earliness < 3h, train/test:')
    #print_mainevals(eval_out['1-3h_trainVsTest_regBkg'], False)
    #print('3h < earliness < 6h, test:')
    #print_mainevals(eval_out['3-6h_regBkg'])
    #print('3h < earliness < 6h, train/test:')
    #print_mainevals(eval_out['3-6h_trainVsTest_regBkg'], False)
    #pprint.pprint(eval_out['stdeval_regBkg'])

# PLOTS
emax_test_hist2d = 1.3e3
makeplots = (not binaryClassif) and True

if makeplots:

    # SHAP values for feature importance
    explainer = shap.Explainer(model_Sig, feature_names=varnames)
    shap_values = explainer(X_test[passClassif_test])
    print()
    print(shap_values)
    print(shap_values.shape)
    print(shap_values.values.shape)
    print(shap_values.data.shape)

    # Aggregate SHAP values by variable or by timerange
    shap_eachvar = copy.deepcopy(shap_values)
    shap_eachrange = copy.deepcopy(shap_values)
    shap_eachrange_coarse = copy.deepcopy(shap_values)

    nvar_tintegrated = len(coarse_timerange)
    varmap_eachvar = []
    varmap_eachrange_coarse = np.empty((n_traintimes_coarse, np.count_nonzero(coarse_timerange)), dtype=int)
    varmap_eachrange = np.empty((n_traintimes, nvar_tintegrated - np.count_nonzero(coarse_timerange)), dtype=int)
    varnames_eachvar = []
    varnames_eachrange = []
    varnames_eachrange_coarse = []

    featidx = 0
    v_coarse = 0
    v_noncoarse = 0
    for v in range(nvar_tintegrated):
        varnames_eachvar.append(varnames[featidx].split('_t-')[0])
        if coarse_timerange[v]:
            varmap_eachvar.append(np.arange(featidx, featidx+n_traintimes_coarse))
            for i in range(n_traintimes_coarse):
                varmap_eachrange_coarse[i][v_coarse] = featidx + i
            if len(varnames_eachrange_coarse) == 0:
                varnames_eachrange_coarse = [ 'time' + varnames[featidx+i].split('_t-')[1] for i in range(n_traintimes_coarse)]
            featidx += n_traintimes_coarse
            v_coarse += 1
        else:
            varmap_eachvar.append(np.arange(featidx, featidx+n_traintimes))
            for i in range(n_traintimes):
                varmap_eachrange[i][v_noncoarse] = featidx + i
            if len(varnames_eachrange) == 0:
                varnames_eachrange = [ 'time' + varnames[featidx+i].split('_t-')[1] for i in range(n_traintimes)]
            featidx += n_traintimes
            v_noncoarse += 1

    # New SHAP values for each variable
    shap_eachvar_vals = np.empty((shap_values.shape[0], nvar_tintegrated))
    shap_eachvar_data = np.empty((shap_values.shape[0], nvar_tintegrated))
    for v in range(nvar_tintegrated):
        shap_eachvar_vals[:, v] = np.sum(shap_values.values[:,varmap_eachvar[v]], axis=1)
        shap_eachvar_data[:, v] = np.mean(shap_values.data[:,varmap_eachvar[v]], axis=1) # this is not correct but is only used to show the feature value in the plots
    shap_eachvar.values = shap_eachvar_vals
    shap_eachvar.data = shap_eachvar_data
    shap_eachvar.feature_names=varnames_eachvar

    # New SHAP values for each time range
    shap_eachrange_vals = np.empty((shap_values.shape[0], n_traintimes))
    shap_eachrange_data = np.empty((shap_values.shape[0], n_traintimes))
    for v in range(n_traintimes):
        shap_eachrange_vals[:, v] = np.sum(shap_values.values[:,varmap_eachrange[v]], axis=1)
        shap_eachrange_data[:, v] = np.mean(shap_values.data[:,varmap_eachrange[v]], axis=1) # this is not correct but is only used to show the feature value in the plots
    shap_eachrange.values = shap_eachrange_vals
    shap_eachrange.data = shap_eachrange_data
    shap_eachrange.feature_names=varnames_eachrange

    # New SHAP values for each coarse time range
    shap_eachrange_coarse_vals = np.empty((shap_values.shape[0], n_traintimes_coarse))
    shap_eachrange_coarse_data = np.empty((shap_values.shape[0], n_traintimes_coarse))
    for v in range(n_traintimes_coarse):
        shap_eachrange_coarse_vals[:, v] = np.sum(shap_values.values[:,varmap_eachrange_coarse[v]], axis=1)
        shap_eachrange_coarse_data[:, v] = np.mean(shap_values.data[:,varmap_eachrange_coarse[v]], axis=1) # this is not correct but is only used to show the feature value in the plots
    shap_eachrange_coarse.values = shap_eachrange_coarse_vals
    shap_eachrange_coarse.data = shap_eachrange_coarse_data
    shap_eachrange_coarse.feature_names=varnames_eachrange_coarse
    print

    plt.figure()
    shap.plots.beeswarm(shap_eachvar, show=False, max_display=40)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_beeswarm_timeIntegrated.pdf'), dpi=250, bbox_inches='tight')

    plt.figure()
    shap.plots.bar(shap_eachvar, show=False, max_display=40)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_bar_timeIntegrated.pdf'), dpi=250, bbox_inches='tight')

    plt.figure()
    shap.plots.beeswarm(shap_eachrange, show=False, max_display=40)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_beeswarm_perTimerange.pdf'), dpi=250, bbox_inches='tight')

    plt.figure()
    shap.plots.bar(shap_eachrange, show=False, max_display=40)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_bar_perTimerange.pdf'), dpi=250, bbox_inches='tight')

    plt.figure()
    shap.plots.beeswarm(shap_eachrange_coarse, show=False, max_display=40)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_beeswarm_perCoarseTimerange.pdf'), dpi=250, bbox_inches='tight')

    plt.figure()
    shap.plots.bar(shap_eachrange_coarse, show=False, max_display=40)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_bar_perCoarseTimerange.pdf'), dpi=250, bbox_inches='tight')

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=40)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_beeswarm.pdf'), dpi=250, bbox_inches='tight')
    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=40) 
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_bar.pdf'), dpi=250, bbox_inches='tight')
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=142)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_beeswarm_allvars.pdf'), dpi=250, bbox_inches='tight')
    #plt.figure()
    #shap.plots.bar(shap_values, show=False, max_display=142)
    #plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_bar_allvars.pdf'), dpi=250, bbox_inches='tight')
    #plt.figure()
    #shap.plots.bar(shap_values[y_test[passClassif_test] < transform_target(6*3600)], show=False, max_display=142)
    #plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_bar_trueEarlinessBelow6h.pdf'), dpi=250, bbox_inches='tight')
    #plt.figure()
    #shap.plots.bar(shap_values[y_test[passClassif_test] >= transform_target(6*3600)], show=False, max_display=142)
    #plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_bar_trueEarlinessAbove6h.pdf'), dpi=250, bbox_inches='tight')
    plt.figure()
    shap.plots.beeswarm(shap_values[y_test[passClassif_test] < transform_target(6*3600)], show=False, max_display=142)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_beeswarm_trueEarlinessBelow6h.pdf'), dpi=250, bbox_inches='tight')
    plt.figure()
    shap.plots.beeswarm(shap_values[y_test[passClassif_test] >= transform_target(6*3600)], show=False, max_display=142)
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAP_beeswarm_trueEarlinessAbove6h.pdf'), dpi=250, bbox_inches='tight')

    exit()

    scatter_true_vs_pred(true_test_regSig, pred_test_regSig, id_idx, test_inds, loge=True)
    scatter_true_vs_pred(true_test_regSig, pred_test_regSig, id_idx, test_inds, loge=False)
    hist_true_vs_pred(true_test_regSig, pred_test_regSig, zmax=emax_test_hist2d, 
                      loge=True, normalize_cols=True, addsavename='_highEweight'+str(wmin))

    # check for overtraining with true vs pred_test for training sample
    scatter_true_vs_pred(true_train_regSig, pred_train_regSig, id_idx, train_inds, loge=True, trainsamples=True)
    hist_true_vs_pred(true_train_regSig, pred_train_regSig, zmax=emax_test_hist2d*((1-test_size)/test_size), 
                      loge=True, trainsamples=True, normalize_cols=True, addsavename='_highEweight'+str(wmin))


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

    # plot correlation between variables
    correlations = np.corrcoef(X_train, rowvar=False)
    correlation_heatmap(correlations, zoom_extremes=True)
