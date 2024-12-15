import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import EvalML as eval
import pickle
import pandas as pd
from matplotlib import colors

def calc_results_per_comp(compoID_idx, true_times, time, predicted_times, id_dict,
                      shap_eachvar,
                      thresholds, 
                      pred_thresholds, 
                      n_bkgmin = 30, 
                      n_sigmin = 4, 
                      sig_max=3600*12,
                      sig_min=0,
                      t_inst_lim = 3000,
                      n_comp_lim = 1000):
    shap_varnames = shap_eachvar.feature_names

    ROCAUC_percomp = np.empty(len(thresholds), dtype=object)
    PRAUC_percomp = np.empty(len(thresholds), dtype=object)
    signalfrac_percomp = np.empty(len(thresholds), dtype=object)
    comp_ids = []
    for i in range(len(thresholds)):
        ROCAUC_percomp[i] = []
        PRAUC_percomp[i] = []
        signalfrac_percomp[i] = []
    TPR_alltrans = np.zeros((len(thresholds), t_inst_lim, len(pred_thresholds)+2))
    FPR_allcomp = np.zeros((len(thresholds), t_inst_lim, len(pred_thresholds)+2)) 
    first_truewarn = np.zeros((n_comp_lim, len(pred_thresholds)+2))
    n_truewarn = np.zeros((len(thresholds), len(pred_thresholds)+2))
    n_falsewarn = np.zeros((len(thresholds), len(pred_thresholds)+2))
    frac_falsewarn = np.zeros((len(thresholds), len(pred_thresholds)))
    n_falsesequences = np.zeros(len(thresholds))
    
    true_per_comp = np.full((n_comp_lim, t_inst_lim), np.nan) 
    time_per_comp = np.full((n_comp_lim, t_inst_lim), np.nan)
    pred_per_comp = np.full((n_comp_lim, t_inst_lim), np.nan)
    allshap_pervar_percomp = np.full((t_inst_lim, n_comp_lim, len(shap_varnames)), np.nan)
    allvari_pervar_percomp = np.full((t_inst_lim, n_comp_lim, len(shap_varnames)), np.nan)
    meanshap_pervar_percomp = np.zeros((t_inst_lim, len(shap_varnames)))
    meanshap_pervar_percomp_sig = np.zeros((t_inst_lim, len(shap_varnames)))
    meanvari_pervar_percomp = np.zeros((t_inst_lim, len(shap_varnames)))
    shap_varrank_percomp_unfilt_sig = np.zeros((t_inst_lim, len(shap_varnames)))
    shap_varrank_percomp_unfilt = np.zeros((t_inst_lim, len(shap_varnames)))
    pred_comp_no_trans = []

    n_sections = np.zeros(len(thresholds)) # number of sequences of length true_threshold in all compositions
    n_trans = 0
    n_comp = 0

    print('# of compositions to start with:', len(np.unique(compoID_idx)))

    for i_id in range(0, np.max(compoID_idx)):
        print('i_id: ' + str(i_id))
        # keep only time instances in this composition
        inds = np.where(compoID_idx == i_id)[0]
        if len(inds) == 0:
            continue
        print('compo id=', id_dict[i_id])

        # separate indices for a composition that has multiple transitions
        true_comp = true_times[inds]
        #print('true_comp: ', true_comp)
        separate = np.where(np.diff(true_comp) > 0.1)[0] + 1
        separate = [0]+list(separate)+[len(inds)]
        #print('separate = ',separate)
        if np.all(true_comp == 43200.008):
            pred_comp_no_trans.append(predicted_times[inds])
            print('no transition in this composition')

        # loop over transitions
        for s in range(1, len(separate)):
            #print('between-transition period #',s, 'inds from to ', separate[s-1], separate[s])
            inds_sep = inds[separate[s-1] : separate[s]]

            # actual data used for this transition/composition
            true_comp = true_times[inds_sep]
            time_comp = time[inds_sep]
            pred_comp = predicted_times[inds_sep]
        
            shapvals_abs = np.abs(shap_eachvar.values[inds_sep])
            is_sig = np.where((true_comp<=sig_max) & (true_comp>=sig_min))[0]
            shapvals_abs_sig = shapvals_abs[is_sig,:]
            
            shapvals = shap_eachvar.values[inds_sep]
            shapvals_sig = shapvals[is_sig,:]
            varivals = shap_eachvar.data[inds_sep]
            varivals_sig = varivals[is_sig,:]

            # FPR for all compositions
            if np.all([ np.count_nonzero(true_comp > thres) >= n_bkgmin for thres in thresholds ]):
                n_comp += 1
                print('n_comp: ' + str(n_comp))
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
            if np.all([ np.count_nonzero(true_comp > thres) >= n_bkgmin for thres in thresholds ]) and \
                np.all([ np.count_nonzero(true_comp < thres) >= n_sigmin for thres in thresholds ]):
                #n_trans += 1
                print('n_trans: ' + str(n_trans))
                print('compo id trans=', id_dict[i_id])
                comp_ids.append(i_id)

                true_per_comp[n_trans,0:len(true_comp)] = true_comp
                time_per_comp[n_trans,0:len(time_comp)] = time_comp
                pred_per_comp[n_trans,0:len(pred_comp)] = pred_comp
                for i, thres in enumerate(list(thresholds)):
                    #print('thres',thres)

                    # TPR values for all transitions
                    # use fixed pred_threshold so that all results are comparable at given pred_threshold
                    eval_everyTrans = eval.EvalML(true_comp, pred_comp, thres, pred_thresholds=pred_thresholds, computeTN=False, force_noReversePredCut=True)
                    eval_everyTrans.set_TPR()
                    TPR_alltrans[i, n_trans] = eval_everyTrans.TPR

                    # ROC and PR calculation for standard EvalML (adaptive pred_thresholds)
                    ev = eval.EvalML(true_comp, pred_comp, thres, n_pred_cuts=100, computeTN=False, force_noReversePredCut=True)
                    ROCAUC_percomp[i].append(ev.calc_ROCAUC())
                    PRAUC_percomp[i].append(ev.calc_PRAUC())
                    signalfrac_percomp[i].append( ev.signal_fraction )
                    
                    # count compositions that issue a true warning
                    for j,predthr in enumerate(pred_thresholds):
                        inds_warn = np.where((true_comp <= thres) & (pred_comp < predthr))[0]
                        # use the fact that transitions were selected to reach the transition time
                        if len(inds_warn) > 0:
                            n_truewarn[i, j] += true_comp[inds_warn[0]] / thres
                            if true_comp[inds_warn[0]] / thres > 1:
                                print('problem with fractional true warning count !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                # compute first true early warning time (need continuous warning until transition)
                for j,predthr in enumerate(pred_thresholds):
                    inds_warn = np.where(pred_comp < predthr)[0]
                    # gets the last sequence of consecutive indices in inds_warn
                    last_seq = np.split(inds_warn, np.where(np.diff(inds_warn) != 1)[0]+1)[-1]
                    if len(last_seq) > 0 and last_seq[-1] == len(pred_comp) - 1:
                        first_truewarn[n_trans, j] = true_comp[last_seq[0]]
                    else:
                        first_truewarn[n_trans, j] = 0

                ## Ranking each variable by SHAP values for this composition/transition
                allshap_pervar_percomp[n_trans,0:shapvals.shape[0],:] = shapvals
                allvari_pervar_percomp[n_trans,0:varivals.shape[0],:] = varivals
                meanshap_pervar_percomp[n_trans] = np.mean(shapvals, axis=0)
                meanshap_pervar_percomp_sig[n_trans] = np.mean(shapvals_sig, axis=0)
                meanvari_pervar_percomp[n_trans] = np.mean(varivals, axis=0)
                shap_varrank_percomp_unfilt[n_trans] = np.argsort(meanshap_pervar_percomp[n_trans])[::-1]
                shap_varrank_percomp_unfilt_sig[n_trans] = np.argsort(meanshap_pervar_percomp_sig[n_trans])[::-1]

                n_trans += 1
            else:
                count_bkg = [np.count_nonzero(true_comp > thres) for thres in thresholds ]
                count_sig = [np.count_nonzero(true_comp < thres) for thres in thresholds ]
                     
    # keep right number of transitions or compositions
    TPR_alltrans = TPR_alltrans[:, 0:n_trans, :]
    FPR_allcomp = FPR_allcomp[:, 0:n_comp, :]
    first_truewarn = first_truewarn[0:n_trans, :]
    frac_truewarn = n_truewarn[:, :-2] / n_trans # fraction of transitions that got a true warning
    for i in range(len(thresholds)):
        frac_falsewarn[i] = n_falsewarn[i][:-2] / n_falsesequences[i]
        ROCAUC_percomp[i] = np.array(ROCAUC_percomp[i])
        PRAUC_percomp[i] = np.array(PRAUC_percomp[i])

    true_per_comp = true_per_comp[0:n_trans]
    time_per_comp = time_per_comp[0:n_trans]
    pred_per_comp = pred_per_comp[0:n_trans]
    meanshap_pervar_percomp = meanshap_pervar_percomp[0:n_trans]
    meanvari_pervar_percomp = meanvari_pervar_percomp[0:n_trans]
    shap_varrank_percomp_unfilt = shap_varrank_percomp_unfilt[0:n_trans]
    allshap_pervar_percomp = allshap_pervar_percomp[0:n_trans]
    allvari_pervar_percomp = allvari_pervar_percomp[0:n_trans]
    
    return [true_per_comp,
            time_per_comp, 
            pred_per_comp,
            meanshap_pervar_percomp,
            meanvari_pervar_percomp,
            allshap_pervar_percomp,
            allvari_pervar_percomp,
            shap_varrank_percomp_unfilt,
            shap_varrank_percomp_unfilt_sig,
            ROCAUC_percomp, PRAUC_percomp,
            signalfrac_percomp, comp_ids, pred_comp_no_trans]


def calc_frac_var_rankings(trans_filter_ROC, shap_varrank_percomp, plot=False):
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

    # count the number of transitions where each variable was ranked first, second, ... in terms of SHAP values
    nvar = len(shap_eachvar.feature_names)
    ntran = len(trans_filter_ROC)
    frac_rankedn = np.zeros((nvar, nvar))
    for vtest in range(nvar):
        frac_rankedn[vtest] = np.count_nonzero(shap_varrank_percomp == vtest, axis=0) 
        #print(shap_varnames[vtest], frac_rankedn[vtest])
    frac_rankedn /= ntran
    # sort by the variables that are most often first
    sortvars = np.lexsort((frac_rankedn[:, 14], frac_rankedn[:, 13], frac_rankedn[:, 12], frac_rankedn[:, 11],
                           frac_rankedn[:, 10], frac_rankedn[:, 9], frac_rankedn[:, 8], frac_rankedn[:, 7], 
                           frac_rankedn[:, 6], frac_rankedn[:, 5], frac_rankedn[:, 4], frac_rankedn[:, 3], 
                           frac_rankedn[:, 2], frac_rankedn[:, 1], frac_rankedn[:, 0]))[::-1]
    frac_rankedn = frac_rankedn[sortvars, :]
    varnames = np.array(varnames)[sortvars]

    if plot:
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
        
    return frac_rankedn


def plot_shap_vs_var(shap_values, 
                     var_values,
                     true_times,
                     sig_max=3600,
                     sig_min=0,
                     inds_times=None,
                     stat_avg='mean',
                     var_percentile=True,
                     xlogbins=False,
                     bin_axis='y',
                     bkg_color=np.array([0.4, 0.6, 0.7]),
                     sig_color=np.array([0.54, 0.05, 0.13]),
                     line_color=np.array([0.54, 0.05, 0.13]),
                     line_alpha = 1,
                     line_color_bkg=np.array([0.3, 0.5, 0.6]),
                     line_width_bkg = 0.,
                     line_width = 1,  
                     err_fill_alpha=0.,
                     data_point_alpha_bkg=0.,
                     data_point_alpha_sig=0.,
                     nbins=100,
                     thin_range=None,
                     line_width_thin=0.5,
                     fold=False):
    
    if inds_times is not None:
        shap_values = shap_values[inds_times] 
        var_values = var_values[inds_times] 
        if true_times is not None:
            true_times = true_times[inds_times]
        
    # get the indices for sorted the true time ascending values
    true_time_sorted_inds = np.argsort(true_times)[::-1]

    #### Make the scatter plot for all the data ####
    y = var_values[true_time_sorted_inds]
    x = shap_values[true_time_sorted_inds]
    c = true_times[true_time_sorted_inds]
    
    sig_bool = (c <= sig_max) & (c>=sig_min)
    is_sig = np.where(sig_bool)[0]
    is_bkg = np.where(~sig_bool)[0]
    #norm = colors.Normalize(vmin=min(c[is_sig]), vmax=max(c[is_sig]))
    
    plt.scatter(x[is_bkg], y[is_bkg], color = bkg_color, s=2, alpha=data_point_alpha_bkg, edgecolors=None)
    #plt.scatter(x[is_sig], y[is_sig], c = norm(c[is_sig]), cmap='viridis', s=2, alpha=data_point_alpha_sig, edgecolors=None)
    plt.scatter(x[is_sig], y[is_sig], color = sig_color, s=2, alpha=data_point_alpha_sig, edgecolors=None)
    
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    
    #plt.figure()
    #sns.histplot(y[is_bkg], color = line_color_bkg, alpha = 0.5, edgecolor=None, label='bkg')
    #sns.histplot(y[is_sig],  color = line_color, alpha = 0.5, edgecolor=None,  label='sig')
    #plt.legend()
    #plt.xlabel('shap value')
    #plt.figure()
    
    ###################################################################################
    #### calculate and plot the mean of the "background", or far-from-transition points
    ###################################################################################
    #meanbkg = scipy.stats.binned_statistic(x[is_bkg], 
    #                                       y[is_bkg], 
    #                                       bins=(np.logspace(np.log10(xmin), 
    #                                                         np.log10(xmax), 
    #                                                         nbins) if xlogbins else np.linspace(xmin, xmax, nbins)), 
    #                                       statistic=stat_avg)

    #stdbkg = scipy.stats.binned_statistic(x[is_bkg], 
    #                                      y[is_bkg], 
    #                                      bins=(np.logspace(np.log10(xmin),
    #                                                        np.log10(xmax), 
    #                                                        nbins) if xlogbins else np.linspace(xmin, xmax, nbins)), 
    #                                      statistic='std')
    #mean_valbkg = meanbkg.statistic
    #mean_binc_bkg = meanbkg.bin_edges
    #mean_binc_bkg = (mean_binc_bkg[:-1] + mean_binc_bkg[1:])/2.

    #plt.plot(mean_binc_bkg, mean_valbkg, color= line_color_bkg, linewidth=line_width_bkg)
    #if line_width_bkg > 0:
    #    plt.fill_between(mean_binc_bkg, 
    #                    mean_valbkg - stdbkg.statistic, 
    #                    mean_valbkg + stdbkg.statistic, 
    #                    edgecolor=None,
    #                    linewidth=0,
    #                    color=line_color_bkg, alpha= err_fill_alpha)

    ###############################################################################
    #### calculate and plot the mean of the "signal", or close-to-transition points
    ###############################################################################
    if var_percentile:
        bins = np.linspace(0, 100, 101)/100.
    else:
        bins = (np.logspace(np.log10(xmin), np.log10(xmax), nbins) if xlogbins else np.linspace(xmin, xmax, nbins))
    if bin_axis=='x':
        bins= (np.logspace(np.log10(xmin), np.log10(xmax), nbins) if xlogbins else np.linspace(xmin, xmax, nbins))
        
        mean = scipy.stats.binned_statistic(x[is_sig], y[is_sig], bins=bins, 
                                            statistic=stat_avg)
        std = scipy.stats.binned_statistic( x[is_sig], y[is_sig], bins=bins, 
                                            statistic='std')
        mean_val = mean.statistic
        mean_binc = mean.bin_edges
        mean_binc = (mean_binc[:-1] + mean_binc[1:])/2.
        plt.plot(mean_binc, mean_val, alpha=line_alpha, linewidth=line_width, color=line_color)
        plt.fill_between(mean_binc, 
                        mean_val - std.statistic, 
                        mean_val + std.statistic, 
                        edgecolor=None,
                        linewidth=0,
                        color=line_color, alpha=err_fill_alpha)
        plt.xlim(1.0*np.nanmax(mean_val), 1.0*np.nanmin(mean_val))
    if bin_axis=='y':
        if var_percentile:
            bins = np.linspace(0, 100, 101)/100.
        else:
            bins= (np.logspace(np.log10(ymin), np.log10(ymax), nbins) if xlogbins else np.linspace(ymin, ymax, nbins))

        # first argument is the values from which bin edges are defined
        # second argumend it the values on which the statistic is calculated. 
        # Here, within the y-axis bin, we find the mean in the x-axis. 
        mean = scipy.stats.binned_statistic(y[is_sig], x[is_sig], bins=bins, 
                                            statistic=stat_avg)
        std = scipy.stats.binned_statistic( y[is_sig], x[is_sig], bins=bins, 
                                            statistic='std')
        mean_val = mean.statistic
        mean_binc = mean.bin_edges
        mean_binc = (mean_binc[:-1] + mean_binc[1:])/2.

        if np.min(mean_binc[~np.isnan(mean_val)]) > 0:
                print('here')
                mean_binc = np.insert(mean_binc, 0, 0)
                print(mean_val[0])
                mean_val = np.insert(mean_val, 0, mean_val[~np.isnan(mean_val)][0])
                print(np.nanmin(mean_val))
       
        if thin_range is not None:
            thin_min = thin_range[0]
            thin_max = thin_range[1]
            thin_mask = (mean_binc >= thin_min) & (mean_binc <= thin_max+0.01) & ~np.isnan(mean_val)
            thick_mask = ((mean_binc < thin_min) | (mean_binc >= thin_max)) & ~np.isnan(mean_val)
            plt.plot(mean_val[thin_mask], mean_binc[thin_mask], alpha=line_alpha, linewidth=line_width_thin, color=line_color)
            plt.plot(mean_val[thick_mask], mean_binc[thick_mask], alpha=line_alpha, linewidth=line_width, color=line_color)
        else:
            mask = ~np.isnan(mean_val)
            print(np.min(mean_binc[mask]))
            if fold:
                plt.plot(mean_val[mask], np.abs(mean_binc[mask]-.5)+.5, alpha=line_alpha, linewidth=line_width, color=line_color)
            else:
                plt.plot(mean_val[mask], mean_binc[mask], alpha=line_alpha, linewidth=line_width, color=line_color)
            #plt.fill_betweenx(mean_binc, 
            #                mean_val - std.statistic, 
            #                mean_val + std.statistic, 
            #                edgecolor=None,
            #                linewidth=0,
            #                color=line_color, alpha=err_fill_alpha)
        #plt.xlim(1.0*np.nanmax(mean_val), 1.0*np.nanmin(mean_val))
    ####################
    #### Aesthetics ####
    ####################
    
    # add a horizontal line at 0 for reference
    #print(np.nanmax(mean_val))
    
    #plt.hlines(0, xmin, xmax, linestyle='-', alpha=0.1, color=[0., 0. , 0.], linewidth=1.5)
    sns.despine()

def calc_sankey(shap_varnames, shap_varrank_percomp, 
                sources_sankey = np.array([]),
                targets_sankey = np.array([]),
                values_sankey = np.array([]),
                target_prev = 'all transitions',
                end_node_prev = 0, rank_i=-1, max_rank=2):
    rank_i+=1
    node_names = np.array(shap_varnames)
    uniq, uniq_inv_inds, counts = np.unique(shap_varrank_percomp[:,rank_i], 
                                            return_inverse=True, return_counts=True)
    #print(shap_varrank_percomp[:,rank_i])
    #print('sources: ' +str(np.repeat(target_prev,len(uniq))))
    
    num_targets = len(uniq)

    # set source 
    sources_sankey = np.concatenate((sources_sankey, np.repeat(target_prev,len(uniq))))
    ranks_sankey = np.repeat(rank_i,len(uniq))

    #print(ranks_sankey)
    # set targets
    targets = np.char.add(node_names[uniq.astype(int)], '_' + str(rank_i))
    #print('targets: ' + str(targets))
    targets_sankey = np.concatenate((targets_sankey, targets))

    # set values
    #sankey_nodes[start_node:end_node, 2] = counts
    #print('counts: ' +str(counts))
    values_sankey = np.concatenate((values_sankey, counts))
    
    for targ_i in range(0, num_targets):
        #print('source: ' + str(uniq[targ_i]))
        # targets become the new source 
        # get the composition nodes with this target
        #print('unig targi: ' +str(uniq[targ_i]))
        comp_inds = np.where(shap_varrank_percomp[:,rank_i]==uniq[targ_i])[0]
        if rank_i >=max_rank:
            #continue
            # if the rank gets too deep, then we want to walk the rank back out while 
            return (sources_sankey, targets_sankey, values_sankey)
        (sources_sankey, 
         targets_sankey, 
         values_sankey) = calc_sankey(shap_varnames, 
                                   shap_varrank_percomp[comp_inds,:], 
                                   target_prev = targets[targ_i],
                                   sources_sankey = sources_sankey,
                                   targets_sankey = targets_sankey,
                                   values_sankey = values_sankey,
                                   rank_i=rank_i,
                                   end_node_prev=end_node_prev,
                                   max_rank=max_rank) 
        
            
    return (sources_sankey, targets_sankey, values_sankey)
    
def calc_binned_ranks(correlation_min, shap_varrank_percomp):

    shap_path = os.path.join(os.getcwd(), 'data', 'correlation_matrix.pickle')
    with open(shap_path, 'rb') as f:
        correlation_matrix_data = pickle.load(f)

    correlation_matrix = correlation_matrix_data[0]
    correlation_matrix_vars = np.array(correlation_matrix_data[1])
    corr_matrix_lower = np.tril(correlation_matrix, k=-1)
    rows, cols = np.where((np.abs(corr_matrix_lower) > correlation_min) & (np.abs(corr_matrix_lower) < 1.0))


    vars_to_bin = np.vstack((correlation_matrix_vars[rows], correlation_matrix_vars[cols]))

    # get the bins
    var_bins = []
    bin_list = []
    for i in range(vars_to_bin.shape[1]):
        # if one element in previous is in the next, then add it to the bin list
        if np.any(np.isin(vars_to_bin[:,i], np.array(bin_list))):
            bin_list.append(vars_to_bin[:,i][0])
            bin_list.append(vars_to_bin[:,i][1])
        else:
            if len(bin_list)>0:
                var_bins.append(np.unique(bin_list))
            bin_list = [vars_to_bin[:,i][0], vars_to_bin[:,i][1]]
        if i==vars_to_bin.shape[1]-1:
            var_bins.append(np.unique(bin_list))

    for j in range(0,len(var_bins)):
        print(var_bins[j])
        print('\n')

    # get the bin or var names
    shap_binvarnames = np.array([])
    var_all_binned = np.array([])
    for j in range(0,len(var_bins)):
        shap_binvarnames = np.concatenate((shap_binvarnames,np.array(['bin' + str(j)])))
        var_all_binned = np.concatenate((var_all_binned, var_bins[j]))

    shap_featnames = np.array(shap_eachvar.feature_names)
    not_in_bins = shap_featnames[~np.isin(shap_featnames, var_all_binned)]
    shap_binvarnames = np.concatenate((shap_binvarnames, not_in_bins))
    print(shap_binvarnames)
    #shap_binvarnames = np.array(['bin0', 'bin1', 'bin2', 'variance2', 'frac_redundant_color_changes', 'frac_attack_changes', 'autocorr'])

    # get the ranking
    ncomps = shap_varrank_percomp.shape[0]
    shap_binvarrank_percomp = np.zeros((ncomps, len(shap_binvarnames)))
    for i in range(1,ncomps):
        varrank_comp = correlation_matrix_vars[shap_varrank_percomp[i,:].astype(int)]
        for j in range(len(varrank_comp)):
            for k in range(len(var_bins)):
                if np.isin(varrank_comp[j],var_bins[k]):
                    varrank_comp[j]= 'bin' + str(k)
        uniq, indices = np.unique(varrank_comp, return_index=True)
        uniq_sort = uniq[np.argsort(indices)]
        index_map = {value: index for index, value in enumerate(shap_binvarnames)}
        shap_binvarrank_percomp[i, :] = np.array([index_map[name] for name in uniq_sort])
        
    return shap_binvarnames, shap_binvarrank_percomp

def plot_slope_corr(shap_varnames, slopes):
    slopes_bin = np.copy(slopes)
    #slopes_bin[slopes_bin<0] = -1
    #slopes_bin[slopes_bin>0] = 1

    slope_data = {
        str(shap_varnames[0]): slopes_bin[:,0],
        str(shap_varnames[1]): slopes_bin[:,1],
        str(shap_varnames[2]): slopes_bin[:,2],
        str(shap_varnames[3]): slopes_bin[:,3],
        str(shap_varnames[4]): slopes_bin[:,4],
        str(shap_varnames[5]): slopes_bin[:,5],
        str(shap_varnames[6]): slopes_bin[:,6],
        str(shap_varnames[7]): slopes_bin[:,7],
        str(shap_varnames[8]): slopes_bin[:,8],
        str(shap_varnames[9]): slopes_bin[:,9],
        str(shap_varnames[10]): slopes_bin[:,10],
        str(shap_varnames[11]): slopes_bin[:,11],
        str(shap_varnames[12]): slopes_bin[:,12],
        str(shap_varnames[13]): slopes_bin[:,13],
        str(shap_varnames[14]): slopes_bin[:,14],
        str(shap_varnames[15]): slopes_bin[:,15],
        str(shap_varnames[16]): slopes_bin[:,16],
        str(shap_varnames[17]): slopes_bin[:,17],
        str(shap_varnames[18]): slopes_bin[:,18]
    }

    df = pd.DataFrame(slope_data)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='RdBu', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)

def var_to_percentile(var_per_comp):
    percentiles = np.arange(0,101,1)
    percentile_bins = np.nanpercentile(var_per_comp, percentiles)
    bin_indices = np.digitize(var_per_comp, percentile_bins)
    x_all_comps = percentiles[bin_indices-1]/100
    return x_all_comps

def calc_slopes(allvari_pervar_percomp_filter, 
                allshap_pervar_percomp_filter, 
                shap_varnames,
                percentiles_x=True,
                max_error=1.):
    slopes = np.zeros((allvari_pervar_percomp_filter.shape[0], allvari_pervar_percomp_filter.shape[2]))
    weights = np.zeros((allvari_pervar_percomp_filter.shape[0], allvari_pervar_percomp_filter.shape[2]))
    
    for v in range(allvari_pervar_percomp_filter.shape[2]):
        if percentiles_x:
            x_all_comps = var_to_percentile(allvari_pervar_percomp_filter[:,:,v])
        else:
            x_max = np.nanpercentile(allvari_pervar_percomp_filter[:,:,v], 90)#np.nanmax(allvari_pervar_percomp_filter[:,:,v])
            x_min = np.nanmin(allvari_pervar_percomp_filter[:,:,v])
            
        for c in range(allvari_pervar_percomp_filter.shape[0]):
            y = allshap_pervar_percomp_filter[c,:,v]
            if percentiles_x:
                vari_x = allvari_pervar_percomp_filter[c,:,v]
                x = x_all_comps[c,:]
                x = x[~np.isnan(vari_x)]
                y = y[~np.isnan(vari_x)]
            else:
                x = allvari_pervar_percomp_filter[c,:,v]
                y = y[~np.isnan(x)]
                x = x[~np.isnan(x)]
                x = (x-x_min)/(x_max-x_min)
            if x.size==0:
                m = np.nan
                continue
            elif np.all(x)==0:
                b=0
                m=0
                y_pred = np.zeros(len(y))
            else:
                p, res, _, _, _ = np.polyfit(x, y, 1, full=True)
                m, b  = p
                y_pred = m*x + b 
            #if v==0:
            #    plt.title(shap_varnames[v])
            #    plt.plot(x, y, '.', alpha=0.)
            #    plt.plot(x, m*x + b, color='k', alpha=0.05)
            slopes[c,v] = m
            if m==np.inf:
                print(m)
            standard_error = np.sqrt(np.sum((y - y_pred)**2) / (len(y) - 2))
            #print(standard_error/np.abs(np.std(y)))
            #print(np.abs(m)/standard_error)

            if standard_error/np.abs(np.std(y)) > max_error:
                weights[c,v] = 0
            else: 
                weights[c,v] = 1
            #weights[c,v] = len(y)**2/np.sum(np.abs(res))
        print(shap_varnames[v])
        print(np.sum(weights[:,v]))
        print('\n')
    return slopes, weights