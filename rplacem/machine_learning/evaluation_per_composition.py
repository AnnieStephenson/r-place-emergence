import numpy as np
import rplacem.globalvariables_peryear as vars
var = vars.var
import EvalML as eval
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

ml_param = eval.AlgoParam(type='regression', # or 'classification
                          test2023=False,
                          n_features=None,
                          num_rounds=None, 
                          learning_rate=0.05, 
                          max_depth=8, 
                          min_child_weight=None, 
                          subsample=0.8, 
                          colsample=0.75, 
                          log_subtract_transform=1.5, 
                          weight_highEarliness=0.2,
                          calibrate_pred=True)
ml_param.num_rounds = 80 if ml_param.test2023 else 140 # Max number of boosting rounds (iterations)
ml_param.min_child_weight = 3.1 if ml_param.type == 'regression' else 4

# Load the npz file
with np.load(os.path.join(var.DATA_PATH, 
                          'earliness_true_vs_predicted_'+ ('regression' if ml_param.type == 'regression' else 'classification')+('_test2023' if ml_param.test2023 else '')+'.npz'),
                          allow_pickle=True) as data:
    true = data['true']
    predicted = data['predicted']
    compoID_idx = data['compoID_idx']
    time = data['time']
    id_dict = data['id_dict']

with open(os.path.join(var.DATA_PATH, 'SHAP_values.pickle'), 'rb') as f:
    shap_res = pickle.load(f)
#shap_res = [shap_values, shap_eachvar, shap_eachrange, shap_eachrange_coarse, inds_onlyStableTimes]
shap_eachvar = shap_res[1]
shap_varnames = shap_eachvar.feature_names
print(shap_eachvar.shape)


# info that we want to plot
thresholds = np.array([1200, 3600, 3*3600])
pred_thresholds = np.concatenate(([0], np.logspace(np.log10(20), np.log10(17*3600), 200)))
ROCAUC_percomp = np.empty(len(thresholds), dtype=object)
PRAUC_percomp = np.empty(len(thresholds), dtype=object)
signalfrac_percomp = np.empty(len(thresholds), dtype=object)
for i in range(len(thresholds)):
    ROCAUC_percomp[i] = []
    PRAUC_percomp[i] = []
    signalfrac_percomp[i] = []
TPR_alltrans = np.zeros((len(thresholds), 1000, len(pred_thresholds)+2)) # 1000 is just an upper limit
FPR_allcomp = np.zeros((len(thresholds), 3000, len(pred_thresholds)+2)) 
first_truewarn = np.zeros((1000, len(pred_thresholds)+2))
n_truewarn = np.zeros((len(thresholds), len(pred_thresholds)+2))
n_falsewarn = np.zeros((len(thresholds), len(pred_thresholds)+2))
frac_falsewarn = np.zeros((len(thresholds), len(pred_thresholds)))
n_falsesequences = np.zeros(len(thresholds))
meanshap_pervar_percomp = np.zeros((1000, len(shap_varnames)))
shap_varrank_percomp = np.zeros((1000, len(shap_varnames)))

n_sections = np.zeros(len(thresholds)) # number of sequences of length true_threshold in all compositions
n_trans = 0
n_comp = 0

print('# of compositions to start with:', len(np.unique(compoID_idx)))

for i_id in range(0, np.max(compoID_idx)):
    # keep only time instances in this composition
    inds = np.where(compoID_idx == i_id)[0]
    if len(inds) == 0:
        continue
    #print('compo id=', id_dict[i_id])
    
    # separate indices for a composition that has multiple transitions
    true_comp = true[inds]
    separate = np.where(np.diff(true_comp) > 0.1)[0] + 1
    separate = [0]+list(separate)+[len(inds)]
    #print('separate = ',separate)

    # loop over transitions
    for s in range(1, len(separate)):
        #print('between-transition period #',s, 'inds from to ', separate[s-1], separate[s])
        inds_sep = inds[separate[s-1] : separate[s]]

        # actual data used for this transition/composition
        true_comp = true[inds_sep]
        time_comp = time[inds_sep]
        pred_comp = predicted[inds_sep]
        shapvals_abs = np.abs(shap_eachvar.values[inds_sep])
    
        bkgmin = 30
        sigmin = 4

        # FPR for all compositions
        if np.all([ np.count_nonzero(true_comp > thres) >= bkgmin for thres in thresholds ]):
            n_comp += 1
            for i, thres in enumerate(list(thresholds)):
                # use fixed pred_threshold so that all results are comparable at given pred_threshold
                eval_everyComp = eval.EvalML(true_comp, pred_comp, thres, pred_thresholds=pred_thresholds, computeTN=False, force_noReversePredCut=True)
                eval_everyComp.set_FPR()
                FPR_allcomp[i, n_comp] = eval_everyComp.FPR

                # cut the available time points in sequences of length thres. A new sequence is started after a cut in time.
                # remove true times
                true_inds = np.where(true_comp < thres)[0]
                falsetimes = np.delete(time_comp, true_inds)
                preds = np.delete(pred_comp, true_inds)
                # make sequences
                split_inds = np.where(np.diff(time_comp) > 301)[0]+1
                continuoustime_sequences = np.split(falsetimes, split_inds)
                pred_sequences = np.split(preds, split_inds)
                for pred_seq, time_seq in zip(pred_sequences, continuoustime_sequences):
                    if np.any(np.diff(time_seq) > 301):
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Problem in separating sequences')
                    # count max number of possible false warnings
                    n_interval = int((thres+0.1) / 300) # number of time intervals within the warning cooldown
                    n_falsesequences[i] += len(pred_seq) / n_interval
                    # count actual number of false warnings
                    for j,predthr in enumerate(pred_thresholds):
                        warn_inds = np.where(pred_seq < predthr)[0]
                        # count warnings using the cooldown
                        last_warn = - n_interval
                        for ind in warn_inds:
                            if ind >= last_warn + n_interval:
                                n_falsewarn[i, j] += 1
                                last_warn = ind

        
        # keep same set of compositions for all thresholds, which limits a bit the available ones
        if np.all([ np.count_nonzero(true_comp > thres) >= bkgmin for thres in thresholds ]) and \
            np.all([ np.count_nonzero(true_comp < thres) >= sigmin for thres in thresholds ]):
            n_trans += 1

            for i, thres in enumerate(list(thresholds)):
                #print('thres',thres)

                # TPR values for all transitions
                # use fixed pred_threshold so that all results are comparable at given pred_threshold
                eval_everyTrans = eval.EvalML(true_comp, pred_comp, thres, pred_thresholds=pred_thresholds, computeTN=False, force_noReversePredCut=True)
                eval_everyTrans.set_TPR()
                TPR_alltrans[i, n_trans] = eval_everyTrans.TPR

                # ROC and PR calculation for standard EvalML (adaptive pred_thresholds)
                ev = eval.EvalML(true_comp, pred_comp, thres, n_pred_cuts=100, computeTN=False, force_noReversePredCut=True)
                ROCAUC_percomp[i].append( ev.calc_ROCAUC() )
                PRAUC_percomp[i].append( ev.calc_PRAUC() )
                signalfrac_percomp[i].append( ev.signal_fraction )
                #print('nsig, nbkg = ', np.count_nonzero(true_comp < thres), np.count_nonzero(true_comp > thres))

                # count compositions that issue a true warning
                for j,predthr in enumerate(pred_thresholds):
                    inds_warn = np.where((true_comp <= thres) & (pred_comp < predthr))[0]
                    # use the fact that transitions were selected to reach the transition time
                    if len(inds_warn) > 0:
                        n_truewarn[i, j] += true_comp[inds_warn[0]] / thres
                        if true_comp[inds_warn[0]] / thres > 1:
                            print('problem with fractional true warning count !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                '''
                if i==0 and ROCAUC_percomp[i][-1] > 0.85 and PRAUC_percomp[i][-1]/signalfrac_percomp[i][-1] < 3:
                    print('compo id=', id_dict[i_id])
                    ev.plotROC('test/ROC_'+str(int(thres))+'_'+id_dict[i_id])
                    ev.plotPR('test/PR_'+str(int(thres))+'_'+id_dict[i_id])
                    print(ROCAUC_percomp[i][-1],PRAUC_percomp[i][-1]/signalfrac_percomp[i][-1],'!!!!!!!!!!!!!!!!!')
                    print('nsig, nbkg = ', np.count_nonzero(true_comp < thres), np.count_nonzero(true_comp > thres))
                    print(ev.TPR, ev.FPR, ev.purity)
                '''
        
            # compute first true early warning time (need continuous warning until transition)
            for j,predthr in enumerate(pred_thresholds):
                inds_warn = np.where(pred_comp < predthr)[0]
                # gets the last sequence of consecutive indices in inds_warn
                last_seq = np.split(inds_warn, np.where(np.diff(inds_warn) != 1)[0]+1)[-1]
                if len(last_seq) > 0 and last_seq[-1] == len(pred_comp) - 1:
                    first_truewarn[n_trans, j] = true_comp[last_seq[0]]
                else:
                    first_truewarn[n_trans, j] = 0

            # Ranking each variable by SHAP values for this composition/transition
            meanshap_pervar_percomp[n_trans] = np.mean(shapvals_abs, axis=0)
            #meanshap_pervar_percomp[n_trans] /= np.sum(meanshap_pervar_percomp[n_trans])
            shap_varrank_percomp[n_trans] = np.argsort(meanshap_pervar_percomp[n_trans])[::-1]

# keep right number of transitions or compositions
TPR_alltrans = TPR_alltrans[:, 0:n_trans, :]
FPR_allcomp = FPR_allcomp[:, 0:n_comp, :]
first_truewarn = first_truewarn[0:n_trans, :]
frac_truewarn = n_truewarn[:, :-2] / n_trans # fraction of transitions that got a true warning
for i in range(len(thresholds)):
    frac_falsewarn[i] = n_falsewarn[i][:-2] / n_falsesequences[i]
meanshap_pervar_percomp = meanshap_pervar_percomp[0:n_trans]
shap_varrank_percomp = shap_varrank_percomp[0:n_trans]
print('# of pre-transition periods, and sum with # no-transition compositions', n_trans, n_comp)

fig = plt.figure(figsize=(5,5), clear=True)
ROCmin = 0.3
NcompAbove0p85 = []
NcompAbove0p95 = []
for i, thres in enumerate(list(thresholds)):
    warn_thres_str = ( str(int(thres/3600.))+'h' if thres/3600. > 0.9 else str(int(thres/60))+'min')
    print('plot',len(ROCAUC_percomp[i]),'transitions for warning <',warn_thres_str) 
    ROCAUC_percomp[i] = np.array(ROCAUC_percomp[i])
    ROCAUC_percomp[i][ ROCAUC_percomp[i] < 0.3 ] = ROCmin
    ROCAUC_percomp[i][ ROCAUC_percomp[i] > 1 ] = 1
    NcompAbove0p85.append( np.count_nonzero( np.array(ROCAUC_percomp[i]) > 0.85 ) )
    NcompAbove0p95.append( np.count_nonzero( np.array(ROCAUC_percomp[i]) > 0.95 ) )
    plt.hist(ROCAUC_percomp[i], bins=np.linspace(ROCmin,1,30), histtype=('bar' if thres==1200 else 'step'), label='time-to-trans < '+warn_thres_str, alpha=(0.5 if thres==1200 else 1))

print('# and fraction of compos with ROC > 0.85:', NcompAbove0p85, np.array(NcompAbove0p85) / np.array([len(ROCAUC_percomp[i]) for i in [0,1,2]]))
print('# and fraction of compos with ROC > 0.95:', NcompAbove0p95, np.array(NcompAbove0p95) / np.array([len(ROCAUC_percomp[i]) for i in [0,1,2]]))

plt.ylabel('# of transitions')
plt.xlabel('area under ROC curve')
plt.xlim([ROCmin,1])
plt.legend(loc='upper left')
plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'ROC_AUC_perComposition.pdf'), dpi=350, bbox_inches='tight')

fig = plt.figure(figsize=(4,4), clear=True)
for i, thres in enumerate(list(thresholds)):
    warn_thres_str = ( str(int(thres/3600.))+'h' if thres/3600. > 0.9 else str(int(thres/60))+'min')
    plt.hist(PRAUC_percomp[i], bins=50, histtype='step', label='time-to-trans < '+warn_thres_str)

plt.ylabel('# of transitions')
plt.xlabel('area under PR curve')
#plt.xlim([0,1])
plt.legend()
plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'PR_AUC_perComposition.pdf'), dpi=350, bbox_inches='tight')

fig = plt.figure(figsize=(5,5), clear=True)
PRratioMax = 1.1 * max(np.max(np.array(PRAUC_percomp[-1])/np.array(signalfrac_percomp[-1])), np.max(np.array(PRAUC_percomp[0])/np.array(signalfrac_percomp[0])))
for i, thres in enumerate(list(thresholds)):
    warn_thres_str = ( str(int(thres/3600.))+'h' if thres/3600. > 0.9 else str(int(thres/60))+'min')
    plt.hist(np.array(PRAUC_percomp[i])/np.array(signalfrac_percomp[i]), 
             bins=np.logspace(np.log(0.5), np.log(PRratioMax), 45), histtype=('bar' if thres==1200 else 'step'), label='time-to-trans < '+warn_thres_str, alpha=(0.5 if thres==1200 else 1))

plt.ylabel('# of transitions')
plt.xlabel('area under PR curve / signal fraction')
plt.xlim([0.5,PRratioMax])
plt.xscale('log')
plt.legend()
plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'PRAUC_over_signalFraction_perComposition.pdf'), dpi=350, bbox_inches='tight')

for i, thres in enumerate(list(thresholds)):
    warn_thres_str = ( str(int(thres/3600.))+'h' if thres/3600. > 0.9 else str(int(thres/60))+'min')
    plt.figure(figsize=(8,8), num=1, clear=True)
    plt.scatter(signalfrac_percomp[i], PRAUC_percomp[i], s=1)
    plt.xlabel('signal fraction')
    plt.ylabel('PR AUC')
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SignalFraction_vs_PRAUC_perComposition_earlinessBelow'+warn_thres_str+'.pdf'), dpi=350, bbox_inches='tight')

    plt.figure(figsize=(8,8), num=1, clear=True)
    plt.scatter(signalfrac_percomp[i], ROCAUC_percomp[i], s=1)
    plt.xlabel('signal fraction')
    plt.ylabel('ROC AUC')
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SignalFraction_vs_ROCAUC_perComposition_earlinessBelow'+warn_thres_str+'.pdf'), dpi=350, bbox_inches='tight')

    plt.figure(figsize=(8,8), num=1, clear=True)
    plt.scatter(np.array(PRAUC_percomp[i])/np.array(signalfrac_percomp[i]), ROCAUC_percomp[i], s=1)
    plt.xlabel('PR AUC / signal frac')
    plt.ylabel('ROC AUC')
    plt.xscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'PRAUC_vs_ROCAUC_perComposition_earlinessBelow'+warn_thres_str+'.pdf'), dpi=350, bbox_inches='tight')

    # plot TPR(all trans) and FPR(all compos) at fixed predicted thresholds   
    FPR_50 = np.quantile(FPR_allcomp[i], 0.5, axis=0)[1:-1]
    FPR_70 = np.quantile(FPR_allcomp[i], 0.7, axis=0)[1:-1]
    FPR_90 = np.quantile(FPR_allcomp[i], 0.9, axis=0)[1:-1]
    FPR_30 = np.quantile(FPR_allcomp[i], 0.3, axis=0)[1:-1]
    FPR_10 = np.quantile(FPR_allcomp[i], 0.1, axis=0)[1:-1]
    TPR_50 = np.quantile(TPR_alltrans[i], 0.5, axis=0)[1:-1]
    TPR_70 = np.quantile(TPR_alltrans[i], 0.7, axis=0)[1:-1]
    TPR_90 = np.quantile(TPR_alltrans[i], 0.9, axis=0)[1:-1]
    TPR_30 = np.quantile(TPR_alltrans[i], 0.3, axis=0)[1:-1]
    TPR_10 = np.quantile(TPR_alltrans[i], 0.1, axis=0)[1:-1]

    plt.figure(figsize=(8,8), num=1, clear=True)
    #for c in range(n_comp):
    #    plt.plot(pred_thresholds, FPR_allcomp[i, c, 1:-1], color='red', linewidth=0.2, label=('FPR (all compositions)' if c==0 else None) )
    #for c in range(n_trans):
    #    plt.plot(pred_thresholds, TPR_alltrans[i, c, 1:-1], color='blue', linewidth=0.2, label=('TPR (all transitions)' if c==0 else None) )
    plt.fill_between(pred_thresholds, FPR_30, FPR_70, facecolor='red', alpha=0.6, edgecolor='none')
    plt.fill_between(pred_thresholds, FPR_10, FPR_90, facecolor='red', alpha=0.2, edgecolor='none')
    plt.fill_between(pred_thresholds, TPR_30, TPR_70, facecolor='blue', alpha=0.6, edgecolor='none')
    plt.fill_between(pred_thresholds, TPR_10, TPR_90, facecolor='blue', alpha=0.2, edgecolor='none')
    plt.plot(pred_thresholds, FPR_50, color='red', label='false positive rate (per composition)')
    plt.plot(pred_thresholds, TPR_50, color='blue', label='true positive rate (per transition)')
    
    plt.xlabel('Upper threshold on predicted time-to-transition [s]')
    plt.ylabel('rate')
    plt.legend(loc='lower right')
    plt.xlim([100,10*3600])
    plt.ylim([0,1])
    #plt.xscale('log')
    plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'TPRFPR_fixedPredictedThresholds_earlinessBelow'+warn_thres_str+'.pdf'), dpi=350, bbox_inches='tight')


# Cap first warning time at 6h max
max_1stwarnt = 6*3600
first_truewarn[first_truewarn > max_1stwarnt] = max_1stwarnt

# Plot first warning time per transition and FPR per composition
firstwarn_50 = np.quantile(first_truewarn, 0.5, axis=0)[1:-1]
firstwarn_70 = np.quantile(first_truewarn, 0.7, axis=0)[1:-1]
firstwarn_90 = np.quantile(first_truewarn, 0.9, axis=0)[1:-1]
firstwarn_30 = np.quantile(first_truewarn, 0.3, axis=0)[1:-1]
firstwarn_10 = np.quantile(first_truewarn, 0.1, axis=0)[1:-1]

fig, ax1 = plt.subplots(figsize=(5,5))
ax2 = ax1.twinx() 

ax1.fill_between(pred_thresholds, FPR_30, FPR_70, facecolor='red', alpha=0.6, edgecolor='none')
ax1.fill_between(pred_thresholds, FPR_10, FPR_90, facecolor='red', alpha=0.2, edgecolor='none')
ax1.plot(pred_thresholds, FPR_50, color='red')
ax2.fill_between(pred_thresholds, firstwarn_30, firstwarn_70, facecolor='blue', alpha=0.5, edgecolor='none')
ax2.fill_between(pred_thresholds, firstwarn_10, firstwarn_90, facecolor='blue', alpha=0.15, edgecolor='none')
ax2.plot(pred_thresholds, firstwarn_50, color='blue')

ax1.set_xlabel('Upper threshold on predicted time-to-transition [s]')
ax1.set_ylabel('false positive rate (per composition)', color='red')
ax1.set_xlim([100,10*3600])
ax1.set_ylim([0,1])
ax1.tick_params(axis='y', labelcolor='red')
#ax1.xscale('log')

ax2.set_ylabel('first early warning time (per transition) [s]', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylim([80,max_1stwarnt])
ax2.set_yscale('log')

fig.savefig(os.path.join(var.FIGS_PATH, 'ML', 'FirstTrueWarnTime_and_FPR_per_composition.pdf'), dpi=350, bbox_inches='tight')



# ROC curve for warning system with cooldown
plt.figure(figsize=(5,5), num=1, clear=True)
for i in range(len(thresholds)):
    warn_thres_str = ( str(int(thresholds[i]/3600.))+'h' if thresholds[i]/3600. > 0.9 else str(int(thresholds[i]/60))+'min')
    plt.plot(frac_falsewarn[i], frac_truewarn[i], label=warn_thres_str+' cool-down')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('false warning rate (with cooldown)')
plt.ylabel('true warning score')
plt.legend(loc='lower right')
plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'ROC_warningWithCooldown.pdf'), dpi=350, bbox_inches='tight')



# shap ranking of variables per composition
varnames = ['fraction of differing pixels (vs reference)',
            'fraction of differing pixels (vs t-1)',
            'fraction of attack changes',
            '# of changes',
            'instability',
            'variance',
            'time in runner-up color',
            '# of used colors (mean)',
            '# of used colors (mean top decile)',
            'auto-correlation',
            'time in attack colors',
            'return time (mean)',
            'return time (mean top decile)',
            '# of users (in window)',
            '# of changes / user (in window)',
            'fraction of new users (vs window)',
            'fraction of redundant changes',
            'entropy',
            'fractal dimension',
            ]
# keep only transitions with ROC>0.85 for time-to-trans<20min
trans_highROC = np.where(ROCAUC_percomp[0] > 0.85)[0]
shap_varrank_percomp = shap_varrank_percomp[trans_highROC]

# count the number of transitions where each variable was ranked first, second, ... in terms of SHAP values
nvar = len(shap_varnames)
ntran = len(trans_highROC)
frac_rankedn = np.zeros((nvar, nvar))
for vtest in range(nvar):
    frac_rankedn[vtest] = np.count_nonzero(shap_varrank_percomp == vtest, axis=0) 
    print(shap_varnames[vtest], frac_rankedn[vtest])
frac_rankedn /= ntran
# sort by the variables that are most often first
sortvars = np.lexsort((frac_rankedn[:, 14], frac_rankedn[:, 13], frac_rankedn[:, 12], frac_rankedn[:, 11],
                       frac_rankedn[:, 10], frac_rankedn[:, 9], frac_rankedn[:, 8], frac_rankedn[:, 7], 
                       frac_rankedn[:, 6], frac_rankedn[:, 5], frac_rankedn[:, 4], frac_rankedn[:, 3], 
                       frac_rankedn[:, 2], frac_rankedn[:, 1], frac_rankedn[:, 0]))[::-1]
frac_rankedn = frac_rankedn[sortvars, :]
varnames = np.array(varnames)[sortvars]

# plot rankings for each variable
category_colors = plt.colormaps['rainbow_r'](np.linspace(0, 1, nvar))
fig, ax = plt.subplots(figsize=(7, 5))
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel('fraction of transitions')

frac_rankedn_cum = frac_rankedn.cumsum(axis=1)
for rank, color in enumerate(category_colors):
    widths = frac_rankedn[:, rank]
    starts = frac_rankedn_cum[:, rank] - widths
    rects = ax.barh(varnames, widths, left=starts, height=0.5, color=color)

cmap = mpl.cm.rainbow_r
norm = mpl.colors.BoundaryNorm(range(1, nvar+1), cmap.N)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax, orientation='vertical', label='rank of variable (best mean(|SHAP|) values)')


plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'SHAPvalues_ranking_pretransition.pdf'), dpi=350, bbox_inches='tight')
