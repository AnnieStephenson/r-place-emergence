import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import rplacem.canvas_part_statistics as stat
import rplacem.compute_variables as comp
import rplacem.variables_rplace2022 as var

def find_all_transitions(keep_idx_compos, ntbins, canparts, v, x, y, z):
    ''' Find and count transitions for compositions of indices keep_idx_compos, using parameters v, x, y ,z '''
    num_comp_with_trans = 0
    num_trans = 0
    num_comp = 0

    for i in keep_idx_compos:
        num_comp += 1
        cpstat = stat.CanvasPartStatistics(canparts[i], n_tbins=400,
                                    n_tbins_trans=ntbins, trans_param=[v, x, y, z],
                                    compute_vars={'stability': 0, 'mean_stability': 0, 'entropy' : 0, 'transitions' : 1, 'attackdefense' : 0},
                                    verbose=False, dont_keep_dir=True)
        n_transi = cpstat.num_transitions
        if n_transi > 0:
            num_comp_with_trans += 1
            num_trans += n_transi

    #print('average number of transitions per composition =', num_trans / num_comp)
    #print('fraction of compositions showing a transition =', num_comp_with_trans / num_comp)
    return(num_trans / num_comp, num_comp_with_trans / num_comp)

# get all compos
file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle') 
with open(file_path, 'rb') as f:
    canparts = pickle.load(f)

## get time-dependent stability for all compos
#file_path_stab = os.path.join(var.DATA_PATH, 'stability_all_canvas_compositions.pickle')

#with open(file_path_stab, 'rb') as handle:
#    [mean_stability, stability_vs_time] = pickle.load(handle)


# Test grid of parameters
cutoffs = np.array([5e-3, 7.5e-3, 1e-2, 1.25e-2, 1.6e-2, 2e-2])
cutoff_stables = np.array([1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3])
num_stables = np.array([3600, 7200, 10800, 14400, 18000])
dist_stables = np.array([3600, 7200, 10800, 14400, 18000])

keep_idx_comps = np.nonzero(np.array([(cp.coords.shape[1] >= 100) for cp in canparts]))[0]
num_comps = len(keep_idx_comps)

vary_cutoffs = np.zeros((len(cutoff_stables), len(cutoffs)))
vary_dist = np.zeros((len(dist_stables), len(num_stables)))

# time ranges
ntbins = 100

for k in keep_idx_comps:
    print('compo #',k)
    cpstat = stat.CanvasPartStatistics(canparts[k], n_tbins=400,
                                    n_tbins_trans=ntbins,
                                    compute_vars={'stability': 0, 'mean_stability': 0, 'entropy' : 0, 'transitions' : 0, 'attackdefense' : 0},
                                    verbose=False, dont_keep_dir=True)

    # Vary first 2 parameters
    num_stab = num_stables[3]
    dist_stab = dist_stables[2]
    for i in range(0, len(cutoffs)):
        #print(i)
        for j in range(0, len(cutoff_stables)):
            #print(j)
            cpstat.transition_param = [cutoffs[i], cutoff_stables[j], num_stab, dist_stab]
            cpstat.search_transitions(canparts[k], 1)
            vary_cutoffs[j, i] += int(cpstat.num_transitions > 0) 

    # Vary last 2 parameters
    cut = cutoffs[2]
    cut_stab = cutoff_stables[2]
    for i in range(0, len(num_stables)):
        #print(i)
        for j in range(0, len(dist_stables)):
            #print(j)
            cpstat.transition_param = [cut, cut_stab, num_stables[i], dist_stables[j]]
            cpstat.search_transitions(canparts[k], 1)
            vary_dist[j, i] += (cpstat.num_transitions > 0)
            #print(vary_dist[j, i])

vary_cutoffs /= num_comps
vary_dist /= num_comps

'''
# Vary first 2 parameters
num_stab = num_stables[3]
dist_stab = dist_stables[2]
for i in range(0, len(cutoffs)):
    print(i)
    for j in range(0, len(cutoff_stables)):
        print(j)
        vary_cutoffs[j, i] = find_all_transitions(keep_idx_comps, ntbins, canparts, cutoffs[i], cutoff_stables[j], num_stab, dist_stab)[1] # num_comp_with_trans / num_comp 
        print(vary_cutoffs[j, i])

# Vary last 2 parameters
cut = cutoffs[2]
cut_stab = cutoff_stables[2]
for i in range(0, len(num_stables)):
    print(i)
    for j in range(0, len(dist_stables)):
        print(j)
        vary_dist[j, i] = find_all_transitions(keep_idx_comps, ntbins, canparts, cut, cut_stab, num_stables[i], dist_stables[j])[1]
        print(vary_dist[j, i])
'''

f1 = plt.figure()
ax1 = plt.axes()
pcm = plt.pcolormesh(cutoffs, cutoff_stables, vary_cutoffs, shading='nearest')
plt.xlabel('lower cutoff for transition region')
plt.ylabel('upper cutoff for stable region')
cbar = plt.colorbar(pcm, label='fraction of compositions showing a transition')
plt.text(0.5, 1.02, 'min length of stable interval = ' + str(num_stab), horizontalalignment='center', transform=ax1.transAxes)
plt.text(0.5, 1.07, 'max dist. between stable and transition = ' + str(dist_stab), horizontalalignment='center', transform=ax1.transAxes)
plt.savefig(os.path.join(var.FIGS_PATH, 'Fraction_of_compos_with_transition_vs_cutoff_parameters.png'), bbox_inches='tight')

f2 = plt.figure()
ax2 = plt.axes()
pcm = plt.pcolormesh(num_stables, dist_stables, vary_dist, shading='nearest')
plt.xlabel('min length of stable interval')
plt.ylabel('max distance between stable and transition regions')
cbar = plt.colorbar(pcm, label='fraction of compositions showing a transition')
plt.text(0.5, 1.02, 'lower cutoff for transition region = ' + str(cut), horizontalalignment='center', transform=ax2.transAxes)
plt.text(0.5, 1.07, 'upper cutoff for stable region = ' + str(cut_stab), horizontalalignment='center', transform=ax2.transAxes)
plt.savefig(os.path.join(var.FIGS_PATH, 'Fraction_of_compos_with_transition_vs_distance_parameters.png'), bbox_inches='tight')