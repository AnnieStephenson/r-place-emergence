import json
import matplotlib.pyplot as plt
from rplacem import var as var
import os
import pickle
import numpy as np
import EvalML as eval

fromregression = True

maxdepth = 8
learningrate = 0.05
subsample = 0.8
colsample = 0.75
minchildw = 3.1 if fromregression else 4
logsub_transform = 1.5

TPR = []
FPR = []
thres = ['20min','1h','3h','6h']#','1-3h','3-6h'
#for wmin in [0.2]:
for thre in thres:
    with open(os.path.join(var.DATA_PATH, 'MLevaluation'
                                        +'_'+ ('regression' if fromregression else 'classification')
                                        +(('_earlinessBelow'+thre) if not fromregression else '')
                                        +'_maxdepth'+str(maxdepth)
                                        +'_minchildweight'+str(minchildw)
                                        +'_subsample'+str(subsample)
                                        +'_colsample'+str(colsample)
                                        +'_logsubtransform'+str(logsub_transform)
                                         +'.pickle'), 'rb') as f:
        evals_regression = pickle.load(f)

for ev in evals_regression:
    if ev.variableName != 'TrainOverTest':
        ev.plotROC()
        ev.plotPR()

'''
thres_lab = ['earliness < 20 min','earliness < 1h','earliness < 3h','earliness < 6h','1h < earliness < 3h','3h < earliness < 6h']


cols = ['red','blue','green','orange','magenta','cyan']
plt.subplots(figsize=(5,5))
#for wm in [0]:
#    for th in range(0,len(TPR[0])):
#        print(FPR[wm][th][1:], TPR[wm][th][1:])
#        plt.plot(FPR[wm][th][1:], TPR[wm][th][1:], label=(thres_lab[th] if wm == 0 else None), linewidth=2, linestyle=(':' if wm==2 else ('--' if wm==1 else '-')), color=cols[th])
for th in range(0,len(TPR)):
    plt.plot(FPR[th][1:], TPR[th][1:], label=thres_lab[th], linewidth=2, linestyle='-', color=cols[th])

plt.xlabel('false positive rate', fontsize=17)
plt.ylabel('true positive rate', fontsize=17)
plt.tick_params(axis="x", labelsize=14)
plt.tick_params(axis="y", labelsize=14)
#plt.legend(fontsize=14)
xmin=3e-4
plt.xlim([xmin,1])#0.35])#[2e-3,0.95*np.max(np.array(FPR[2]))])
plt.ylim([0,1])#0.91])
xlog=True
if xlog:
    plt.xscale('log')
plt.plot(np.logspace(np.log10(xmin),1,200) if xlog else np.linspace(0,1,2), np.logspace(np.log10(xmin),1,200) if xlog else np.linspace(0,1,2), color='black', linestyle=':', linewidth=2, label='random classification')
plt.legend()

#plt.text(0.07, 0.79, thres_lab[0], horizontalalignment='left', color=cols[0], fontsize=14, rotation=11)
#plt.text(0.09, 0.64, thres_lab[1], horizontalalignment='left', color=cols[1], fontsize=14, rotation=18)
#plt.text(0.09, 0.49, thres_lab[2], horizontalalignment='left', color=cols[2], fontsize=14, rotation=25)
#plt.text(0.12, 0.4, thres_lab[3], horizontalalignment='left', color=cols[3], fontsize=14, rotation=26)

plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'earliness_ROCcurve.pdf'), bbox_inches='tight')

for i in range(0, len(thres)):
    print('earliness < '+thres[i]+': FPR for TPR=0.5 = ', np.interp(0.5, np.flip(TPR[i][1:]), np.flip(FPR[i][1:])))
'''

# ROC CURVES FOR KENDALL TAU

# extract data from file
file_path = os.path.join(var.DATA_PATH, 'training_data_330variables.pickle')
with open(file_path, 'rb') as f:
    [inputvals, outputval, varnames, eventtime, id_idx, id_dict,
     coarse_timerange, kendall_tau,
     n_traintimes, n_traintimes_coarse
     ] = pickle.load(f) 
    
i_variable = []
for ivar, coarse in enumerate(coarse_timerange):
    start = i_variable[-1] if len(i_variable)>0 else 0
    i_variable += [ivar]*(n_traintimes_coarse if coarse else n_traintimes)

wanted_vars = [('returntime_mean_t-0-0', 0),
               ('returntime_mean_t-0-0', 1),
               ('variance_t-0-0', 0),
               ('variance_t-0-0', 1),
               ('variance2_t-0-0', 1),
               ('autocorr_t-0-0', 0),
               ('autocorr_t-0-0', 1),
               ('autocorr2_t-0-0', 1),
               ('instability_mean_t-0-0', 0),
               ('instability_mean_t-0-0', 1),
               ('frac_pixdiff_inst_vs_swref_t-0-0', 0),
               ]
vars_kendall = []
print(wanted_vars)
for i, vn in enumerate(varnames):
    iskendall = kendall_tau[i_variable[i]]
    if (vn, iskendall) in wanted_vars:
        vars_kendall.append(i)
inputvals = inputvals[:, vars_kendall]
varnames = varnames[vars_kendall]
kendall_tau = np.array(kendall_tau)[np.array(i_variable)[vars_kendall]]

evals_all = []
# loop over variables
for vidx in range(inputvals.shape[1]):
    ktau = inputvals[:, vidx]
    keepinds = np.where((~np.isnan(ktau)))# & (np.abs(ktau) < 1e10))
    # loop over earliness thresholds to test
    for i, th in enumerate([1200, 3600, 3*3600, 6*3600]):
        print(vidx, varnames[vidx], th, 'kendall', kendall_tau[vidx])
        evalROC = eval.EvalML(true=outputval[keepinds], predicted=ktau[keepinds], true_threshold=th, n_pred_cuts=200,
                              variableName=varnames[vidx][:-6]+('_kendallTau' if kendall_tau[vidx] else ''),
                              algo_param=eval.AlgoParam(type='continuous'))
        evalROC.compute_all()

        evalROC.plotROC()
        evalROC.plotPR()
        evals_all.append(evalROC)

with open(os.path.join(var.DATA_PATH, 'MLevaluation_singleVariableNoML.pickle'), 'wb') as f:
    pickle.dump(evals_all, f, protocol=pickle.HIGHEST_PROTOCOL)
