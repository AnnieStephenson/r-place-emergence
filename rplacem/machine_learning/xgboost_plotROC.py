import json
import matplotlib.pyplot as plt
import rplacem.variables_rplace2022 as var
import os
import numpy as np

maxdepth = 7
learningrate = 0.12
subsample = 0.85
colsample = 0.8
minchildw = 2.
logsub_transform = 1.5

TPR = []
FPR = []
thres = ['20min','1h','3h','6h']
for wmin in [0.05,0.1,0.2]:
    with open(os.path.join(var.DATA_PATH, 'MLevaluation_weight-high-e-0p'+str(int(wmin*100))
                                         +'_maxdepth'+str(maxdepth)
                                         +'_minchildweight'+str(minchildw)
                                         +'_subsample'+str(subsample)
                                         +'_colsample'+str(colsample)
                                         +'_logsubtransform'+str(logsub_transform)
                                         +'.json'), 'r') as f:
        eval = json.load(f)

    thres_lab = ['earlyness < 20 min','earlyness < 1h','earlyness < 3h','earlyness < 6h']
    TPR.append([eval[th]['TPR'] for th in thres])
    FPR.append([eval[th]['FPR'] for th in thres])

cols = ['red','blue','green','orange']
plt.subplots(figsize=(5,5))
for wm in [2]:
    for th in range(0,len(TPR[0])):
        print(FPR[wm][th][1:], TPR[wm][th][1:])
        plt.plot(FPR[wm][th][1:], TPR[wm][th][1:], label=(thres_lab[th] if wm == 2 else None), linewidth=2, linestyle=(':' if wm==0 else ('--' if wm==1 else '-')), color=cols[th])
plt.xlabel('false positive rate', fontsize=17)
plt.ylabel('true positive rate', fontsize=17)
plt.tick_params(axis="x", labelsize=14)
plt.tick_params(axis="y", labelsize=14)
#plt.legend(fontsize=14)
plt.xlim([0,0.35])#[2e-3,0.95*np.max(np.array(FPR[2]))])
plt.ylim([0,0.91])
#plt.xscale('log')
plt.plot(np.linspace(0,1,2), np.linspace(0,1,2), color='black', linestyle=':', linewidth=2)

plt.text(0.07, 0.8, thres_lab[0], horizontalalignment='left', color=cols[0], fontsize=14, rotation=8)
plt.text(0.09, 0.64, thres_lab[1], horizontalalignment='left', color=cols[1], fontsize=14, rotation=18)
plt.text(0.09, 0.47, thres_lab[2], horizontalalignment='left', color=cols[2], fontsize=14, rotation=25)
plt.text(0.1, 0.31, thres_lab[3], horizontalalignment='left', color=cols[3], fontsize=14, rotation=31)

plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'earlyness_ROCcurve.pdf'), bbox_inches='tight')

for i in range(0, len(thres)):
    print('earlyness < '+thres[i]+': FPR for TPR=0.5 = ', np.interp(0.5, TPR[2][i][1:], FPR[2][i][1:]))