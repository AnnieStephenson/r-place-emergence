import numpy as np
import xgboost as xg
import os
import math
import pickle
import matplotlib.colors as colors
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import rplacem.variables_rplace2022 as var
import pandas as pd
import json
import pprint
from matplotlib.patches import Rectangle

def scatter_true_vs_pred(true, pred, id_idx, test_inds, loge=True, onlysomecompos=False, trainsamples=False, addsavename=''):

    fig = plt.figure()
    sns.despine()

    plt.ylabel(r'100 + $\bf{predicted}$ earlyness [s]', fontsize=12)
    plt.xlabel(r'100 + $\bf{true}$ earlyness [s]', fontsize=12)
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

    plt.scatter(100. + true[select],
                100. + pred[select],
                c=id_idx[test_inds][select], 
                cmap='rainbow',
                s=1)
    
    plt.plot(np.linspace(emin,emax,10), np.linspace(emin,emax,10), color='black', linestyle='dashed', linewidth=0.5)
    plt.grid(color='black', linewidth=0.3, linestyle='dotted')

    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', ('log_' if loge else '') + 'earlyness_pred_vs_true_scatter'+('_trainingsamples' if trainsamples else '')+addsavename+'.pdf'), 
                dpi=250, bbox_inches='tight')


def hist_true_vs_pred(true, pred, zmax=2e3, 
                      loge=True, 
                      trainsamples=False, normalize_cols=True, addsavename=''):

    fig = plt.figure()
    sns.despine()

    plt.ylabel(r'100 + $\bf{predicted}$ earlyness [s]', fontsize=16)
    plt.xlabel(r'100 + $\bf{true}$ earlyness [s]', fontsize=16)
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
    vm = max(1e-3, np.nanmin(H))
    vM = min(np.nanmax(H), zmax)
    print(vm,vM)
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
    
    plt.colorbar(label=('probability (at given true earlyness)' if normalize_cols else 'counts'), pad=0.02)

    xminrect = 1.1*6*3600
    plt.gca().add_patch(Rectangle((xminrect, 100), emax-xminrect, emax-100, alpha=0.75, color='white', linewidth=0))
    plt.text(7e4, 1000, r'$\bf{not\ \ predictable}$', color='green', fontsize=14, rotation=90)
    
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', ('log_' if loge else '') + 'earlyness_pred_vs_true_hist'+('_trainingsamples' if trainsamples else '')+addsavename+'.pdf'), 
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

# transformation of researched output
logsub_transform = 1.5
def transform_target(targ):
    return np.log10(100. + targ) - logsub_transform

def invtransform_target(targ):
    return np.power(10, targ + logsub_transform) - 100.

# Define the custom objective function
# reverse pred and dtrain args when using sklearn
def squaredrelerr(pred, dtrain): # objective fct = ((pred - true)/true)**2
    true = dtrain.get_label()
    weight = dtrain.get_weight()
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









# extract training data from file
file_path = os.path.join(var.DATA_PATH, 'training_data.pickle')
with open(file_path, 'rb') as f:
    [inputvals, outputval, varnames, eventtime, id_idx, id_dict] = pickle.load(f)

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
wmin = 0.2
xbreak1 = 2e4
xbreak2 = 1e5
weights[outputval_orig <= xbreak1] = wmax
weights[outputval_orig >= xbreak2] = wmin
# power-law interpolation between these two limits
powerexp = math.log(wmax / wmin) / math.log(xbreak1 / xbreak2)
powerlaw_inds = np.where((outputval_orig > xbreak1) & (outputval_orig < xbreak2))
weights[powerlaw_inds] = wmax * np.power(outputval_orig[powerlaw_inds] / xbreak1, powerexp)
sumweights = np.sum(weights)

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
dtrain = xg.DMatrix(data = X_train, label = y_train, weight=w_train, feature_names=varnames)
dtest  = xg.DMatrix(data = X_test,  label = y_test,  weight=w_test,  feature_names=varnames)





# Set XGBoost regressor parameters
num_rounds = 170  # Max number of boosting rounds (iterations)
maxdepth = 7
learningrate = 0.12
subsample = 0.85
colsample = 0.8
minchildw = 2.
params = {
    #'objective': 'reg:squarederror',  # Regression task with squared loss # or: squarederror, squaredlogerror, absoluteerror # this should be 'objective' rather than 'obj' for sklearn
    'max_depth': maxdepth,                    # Maximum depth of each tree
    'learning_rate': learningrate,              # Learning rate
    'subsample': subsample,                  # Subsample ratio of the training instances
    'colsample_bytree': colsample,           # Subsample ratio of features at each new tree
    'colsample_bylevel': colsample,          # Subsample ratio of features at each tree level
    'min_child_weight': minchildw*700000/sumweights,             # minimum sum of event weights (times the hessian value) per node
    'random_state': 42,                # Random seed
    #'eval_metric': ['mae','mphe','rmsle','rmse','mape']#meanrelerr, rmsrelerr, 
    #'verbosity' : 2
}
#results: Dict[str, Dict[str, List[float]]] = {}

# Train the model
print('start training')
eval_res = {}
model = xg.train(params, dtrain, 
                 num_boost_round=num_rounds,
                 early_stopping_rounds=30,
                 obj=squaredrelerr,
                 custom_metric=negF1_3h,
                 evals=[(dtest, 'test')],
                 evals_result=eval_res
                )

#model = xg.XGBRegressor(**params, n_estimators=num_rounds)
#model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
#          sample_weight=w_train, 
#          #eval_sample_weight=[w_test],
#          eval_metric=rmsrelerr, 
#          verbose=True
#          )
print('done training')

# Evaluate the model on the testing set
predictions = model.predict(dtest)
predictions_train = model.predict(dtrain)








# EVALUATION
# weights are in general ignored here

def evals(true, pred, e_thres):
    TPR = []
    FPR = []
    purity = []
    F1 = []
    thres_list = [e_thres, 600, 1200, 2400, 3600, 1.5*3600, 2*3600, 3*3600, 4*3600, 5*3600, 6*3600, 7*3600, 8*3600, 10*3600]
    for i, pred_thres in enumerate(thres_list):
        TPR.append(np.count_nonzero((true < e_thres) & (pred < pred_thres)) / np.count_nonzero(true < e_thres))
        FPR.append(np.count_nonzero((true > e_thres) & (pred < pred_thres)) / np.count_nonzero(true > e_thres))
        purity.append(np.count_nonzero((true < e_thres) & (pred < pred_thres)) / np.count_nonzero(pred < pred_thres))
        F1.append(2*purity[i]*TPR[i] / (purity[i]+TPR[i]))
    frequency = np.count_nonzero(true < e_thres) / np.count_nonzero(true)
    return {'TPR':TPR, 'FPR':FPR, 'purity':purity, 'F1':F1, 'threshold_list':thres_list, 'frequency':frequency}

def print_mainevals(eval, printFreq=True):
    print('             TPR =','%.4f' % eval['TPR'][0], ', FPR =','%.4f' % eval['FPR'][0], ', purity =','%.4f' % eval['purity'][0], ', F1 = ','%.4f' % eval['F1'][0], (', frequency = '+'%.4f' % eval['frequency'] if printFreq else ''))

def divide_evalres(eval1, eval2):
    newTPR = list(np.array(eval1['TPR']) / np.array(eval2['TPR']))
    newFPR = list(np.array(eval1['FPR']) / np.array(eval2['FPR']))
    newpurity = list(np.array(eval1['purity']) / np.array(eval2['purity']))
    newF1 = list(np.array(eval1['F1']) / np.array(eval2['F1']))
    newfrequency = eval1['frequency'] / eval2['frequency']
    return {'TPR':newTPR, 'FPR':newFPR, 'purity':newpurity, 'F1':newF1, 'threshold_list':eval1['threshold_list'], 'frequency':newfrequency}

# return earlyness from its transformation used in training
true_test = invtransform_target(y_test)
pred_test = invtransform_target(predictions)
true_train = invtransform_target(y_train)
pred_train = invtransform_target(predictions_train)


# write evaluation results in a json
with open(os.path.join(var.DATA_PATH, 'MLevaluation_weight-high-e-0p'+str(int(wmin*100))
                                     +'_maxdepth'+str(maxdepth)
                                     +'_minchildweight'+str(minchildw)
                                     +'_subsample'+str(subsample)
                                     +'_colsample'+str(colsample)
                                     +'_logsubtransform'+str(logsub_transform)
                                     +'.json'), 'w') as f:
    eval_out = {'20min': evals(true_test, pred_test, 1200),
               '1h': evals(true_test, pred_test, 3600),
               '3h': evals(true_test, pred_test, 3*3600),
               '6h': evals(true_test, pred_test, 6*3600),
               '20min_train': evals(true_train, pred_train, 1200),
               '1h_train': evals(true_train, pred_train, 3600),
               '3h_train': evals(true_train, pred_train, 3*3600),
               '6h_train': evals(true_train, pred_train, 6*3600),
               'stdeval': dict(zip(eval_res['test'].keys(), list(eval_res['test'].values())[-1]))
               }
    eval_out['20min_trainVsTest'] = divide_evalres(eval_out['20min_train'], eval_out['20min'])
    eval_out['1h_trainVsTest'] = divide_evalres(eval_out['1h_train'], eval_out['1h'])
    eval_out['3h_trainVsTest'] = divide_evalres(eval_out['3h_train'], eval_out['3h'])
    eval_out['6h_trainVsTest'] = divide_evalres(eval_out['6h_train'], eval_out['6h'])
    json.dump(eval_out, f)
    #pprint.pprint(eval_out)
    print('earlyness < 20min, test:')
    print_mainevals(eval_out['20min'])
    print('earlyness < 20min, train/test:')
    print_mainevals(eval_out['20min_trainVsTest'], False)
    print('earlyness < 1h, test:')
    print_mainevals(eval_out['1h'])
    print('earlyness < 1h, train/test:')
    print_mainevals(eval_out['1h_trainVsTest'], False)
    print('earlyness < 3h, test:')
    print_mainevals(eval_out['3h'])
    print('earlyness < 3h, train/test:')
    print_mainevals(eval_out['3h_trainVsTest'], False)
    print('earlyness < 6h, test:')
    print_mainevals(eval_out['6h'])
    print('earlyness < 6h, train/test:')
    print_mainevals(eval_out['6h_trainVsTest'], False)

    pprint.pprint(eval_out['stdeval'])

# PLOTS
emax_test_hist2d = 1.3e3
makeplots = True

if makeplots:
    scatter_true_vs_pred(true_test, pred_test, id_idx, test_inds, loge=True)
    scatter_true_vs_pred(true_test, pred_test, id_idx, test_inds, loge=False)
    hist_true_vs_pred(true_test, pred_test, zmax=emax_test_hist2d, 
                      loge=True, normalize_cols=True, addsavename='_highEweight'+str(wmin))

    # check for overtraining with true vs pred_test for training sample
    scatter_true_vs_pred(true_train, pred_train, id_idx, train_inds, loge=True, trainsamples=True)
    hist_true_vs_pred(true_train, pred_train, zmax=emax_test_hist2d*((1-test_size)/test_size), 
                      loge=True, trainsamples=True, normalize_cols=True, addsavename='_highEweight'+str(wmin))

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

    # plot correlation between variables
    correlations = np.corrcoef(X_train, rowvar=False)
    correlation_heatmap(correlations, zoom_extremes=True)
