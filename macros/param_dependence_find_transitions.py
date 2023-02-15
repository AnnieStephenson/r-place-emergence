import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import rplacem.compute_variables as comp
import rplacem.variables_rplace2022 as var

def find_all_transitions(keep_idx_compos, stability_vs_time, time_ranges, v, x, y, z):
    ''' Find and count transitions for compositions of indices keep_idx_compos, using parameters v, x, y ,z '''
    num_comp_with_trans = 0
    num_trans = 0
    num_comp = 0

    for i in keep_idx_compos:
        num_comp += 1
        transitions = comp.find_transitions(time_ranges, stability_vs_time[i][0], v, x, y ,z)
        if len(transitions[0]) > 0:
            num_comp_with_trans += 1
            num_trans += len(transitions[0])

    #print('average number of transitions per composition =', num_trans / num_comp)
    #print('fraction of compositions showing a transition =', num_comp_with_trans / num_comp)
    return(num_trans / num_comp, num_comp_with_trans / num_comp)

# get all compos
file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle') 
with open(file_path, 'rb') as f:
    canparts = pickle.load(f)

# get time-dependent stability for all compos
file_path_stab = os.path.join(var.DATA_PATH, 'stability_all_canvas_compositions.pickle')

with open(file_path_stab, 'rb') as handle:
    [mean_stability, stability_vs_time] = pickle.load(handle)

# Test grid of parameters
cutoffs = np.array([0.8, 0.83, 0.85, 0.87, 0.885, 0.9, 0.915, 0.93, 0.945, 0.96])
cutoff_stables = np.array([0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 0.998])
num_stables = np.array([1, 2, 3, 4, 5, 6])
dist_stables = np.array([1, 2, 3, 4, 5])

keep_idx_comps = np.nonzero(np.array([(cp.coords.shape[1] >= 100) for cp in canparts]))[0]

vary_cutoffs = np.zeros((len(cutoff_stables), len(cutoffs)))
vary_dist = np.zeros((len(dist_stables), len(num_stables)))

# time ranges
time_bins = 80
time_interval = var.TIME_WHITEONLY / time_bins  # seconds
time_ranges = np.arange(0, var.TIME_WHITEONLY+time_interval-1e-4, time_interval)

# Vary first 2 parameters
num_stab = num_stables[3]
dist_stab = dist_stables[2]
for i in range(0, len(cutoffs)):
    print(i)
    for j in range(0, len(cutoff_stables)):
        print(j)
        vary_cutoffs[j, i] = find_all_transitions(keep_idx_comps, stability_vs_time, time_ranges, cutoffs[i], cutoff_stables[j], num_stab, dist_stab)[1] # num_comp_with_trans / num_comp 


# Vary last 2 parameters
cut = cutoffs[3]
cut_stab = cutoff_stables[4]
for i in range(0, len(num_stables)):
    print(i)
    for j in range(0, len(dist_stables)):
        print(j)
        vary_dist[j, i] = find_all_transitions(keep_idx_comps, stability_vs_time, time_ranges, cut, cut_stab, num_stables[i], dist_stables[j])[1]

f1 = plt.figure()
ax1 = plt.axes()
pcm = plt.pcolormesh(cutoffs, cutoff_stables, vary_cutoffs, shading='nearest')
plt.xlabel('upper cutoff for transition region')
plt.ylabel('lower cutoff for stable region')
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
plt.text(0.5, 1.02, 'upper cutoff for transition region = ' + str(cut), horizontalalignment='center', transform=ax2.transAxes)
plt.text(0.5, 1.07, 'lower cutoff for stable region = ' + str(cut_stab), horizontalalignment='center', transform=ax2.transAxes)
plt.savefig(os.path.join(var.FIGS_PATH, 'Fraction_of_compos_with_transition_vs_distance_parameters.png'), bbox_inches='tight')