import numpy as np
import os
import canvas_part as cp
import thermo as th
import Variables.Variables as var
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def find_all_transitions(keep_idx_compos, stability_vs_time, time_ranges, v, x, y, z):
    ''' Find and count transitions for compositions of indices keep_idx_compos, using parameters v, x, y ,z '''
    num_comp_with_trans = 0
    num_trans = 0
    num_comp = 0

    for i in keep_idx_compos:
        num_comp += 1
        transitions = th.find_transitions(time_ranges, stability_vs_time[i][0], v, x, y ,z)
        if len(transitions[0]) > 0:
            num_comp_with_trans += 1
            num_trans += len(transitions[0])

    #print('average number of transitions per composition =', num_trans / num_comp)
    #print('fraction of compositions showing a transition =', num_comp_with_trans / num_comp)
    return(num_trans / num_comp, num_comp_with_trans / num_comp)

def transition_reference_image(canvas_part, 
                               time_ranges, stability_vs_time,
                               save_images,
                               cutoff=0.88,
                               cutoff_stable=0.985,
                               len_stable_intervals=3,
                               dist_stableregion_transition=3
                               ):
    ''' Finds transitions for a given canvas part, 
    and returns the images containing the most stable pixels for the stable periods before and after the transition, and during the transition.
    stability_vs_time input must be the one corresponding to the input canvas_part.'''
    
    transitions = th.find_transitions(time_ranges, stability_vs_time[0], 
                                      cutoff, cutoff_stable, len_stable_intervals, dist_stableregion_transition)

    avimage_pre = []
    avimage_trans = []
    avimage_post = []
    frac_differing_pixels = []
    trans_times_mod = []

    for j in range(0, len(transitions[0])):
        print(transitions[0][j])
        print(transitions[1][j])
        trans_times2 = np.hstack((0, transitions[1][j]))
        print(trans_times2)
        # number of time intervals used for averaging the pre- and post-transition stable periods
        averaging_period = 1 #len_stable
        trans_times2[1] = trans_times2[2] - averaging_period * (time_ranges[-3] - time_ranges[-4]) # calculate the (pre)stable image in only the latest stable time interval 
        trans_times2[6] = trans_times2[5] + averaging_period * (time_ranges[-3] - time_ranges[-4]) # calculate the (post)stable image in only the earliest stable time interval 
        print(trans_times2)
        trans_times_mod.append(trans_times2)
        _, stablepixels1, stablepixels2, stablepixels3 = th.stability(canvas_part, trans_times2, True, save_images, False, False)
        avimage_pre.append(stablepixels1[1])
        avimage_trans.append(stablepixels1[3])
        avimage_post.append(stablepixels1[5])

        #'pixels' are filled only for the canvas_part.coords, white otherwise. So the differences will show only for the canvas_part.coords
        num_differing_pixels = th.count_image_differences(avimage_post[j], avimage_pre[j], canvas_part)
        frac_differing_pixels.append( num_differing_pixels / len(canvas_part.coords[0]) )
        print(num_differing_pixels, frac_differing_pixels)

    #print('average number of transitions per composition =', num_trans / num_comp)
    #print('fraction of compositions showing a transition =', num_comp_with_trans / num_comp)
    return (avimage_pre, avimage_trans, avimage_post, transitions[1], trans_times_mod)



# get time-dependent stability for all compos
file_path_stab = os.path.join(var.DATA_PATH, 'stability_all_canvas_compositions.pickle')

with open(file_path_stab, 'rb') as handle:
    [mean_stability,stability_vs_time] = pickle.load(handle)

# get all compos
file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle') 
with open(file_path, 'rb') as f:
    canvas_parts = pickle.load(f)

# test only compositions of significant size
#keep_idx_comps = np.nonzero(np.array([(cp.coords.shape[1] >= 100) for cp in canvas_parts]))[0]

time_bins_stab = 80
time_bins_trans = 250
time_interval_stab = var.TIME_WHITEONLY / time_bins_stab  # seconds
time_interval_trans = var.TIME_WHITEONLY / time_bins_trans  # seconds
time_ranges_stab = np.arange(0, var.TIME_WHITEONLY+time_interval_stab-1e-4, time_interval_stab)
time_ranges_trans = np.arange(0, var.TIME_WHITEONLY+time_interval_trans-1e-4, time_interval_trans)

j = 0
#find_all_transitions(keep_idx_comps, stability_vs_time, time_ranges_stab, canvas_parts, 0.88, 0.985, 3, 3, True)
image_pre, _, _, trans_times, trans_times_modif = transition_reference_image(canvas_parts[5], time_ranges_stab, stability_vs_time[5], True, 0.88, 0.99, 3, 4)
res = th.num_deviating_pixels(canvas_parts[5], time_ranges_trans, image_pre[j])
pix_changes = res[0] * 300 / (time_interval_trans * res[3]) 
defense_changes = res[1] * 300 / (time_interval_trans * res[3]) 
attack_changes = res[2] * 300 / (time_interval_trans * res[3])
deviating_pixels = res[4] / res[3]
instability = (1-stability_vs_time[5][0]) * 3600 / time_interval_trans

plt.figure()
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, pix_changes, label='# pixel changes / active area / 5 min')
att, = plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, attack_changes, label='# attack changes / active area / 5 min')
defe, = plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, defense_changes, label='# defense changes / active area / 5 min')
plt.plot(time_ranges_stab[:-1]+time_interval_stab/2, instability, label='(1 - stability) / 1 h')
sns.despine()
plt.xlabel('Time [s]')
ymax = max(pix_changes)*1.4
plt.ylim([0, ymax])
plt.xlim([trans_times[j][0], trans_times[j][5]])
#plt.xlim([trans_times_modif[j][1], trans_times_modif[j][6]])
lege = plt.legend(loc="upper left")
plt.vlines(x = [trans_times_modif[j][1], trans_times_modif[j][2]], ymin=0, ymax=100, colors = 'black', linestyle='dashed')
refimstr = plt.text(trans_times_modif[j][1] + (trans_times_modif[j][2]-trans_times_modif[j][1])*0.6, 0.6*ymax, 'ref image', horizontalalignment='center', verticalalignment='center', rotation=90)
plt.savefig(os.path.join(var.FIGS_PATH, 'history_' + canvas_parts[5].out_name(), 'number_attack_pixel_changes.png'), bbox_inches='tight')


with np.errstate(divide='ignore', invalid='ignore'):
    attack_defense_ratio = attack_changes / defense_changes
attack_defense_ratio[np.where(defense_changes == 0)] = 0.
plt.figure()
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, attack_defense_ratio, label='# attack / # defense changes')
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, pix_changes, label='# pixel changes / active area / 5 min')
plt.plot(time_ranges_stab[:-1]+time_interval_stab/2, instability, label='(1 - stability) / 1 h')
sns.despine()
plt.xlabel('Time [s]')
ymax = max(attack_defense_ratio)*1.3
plt.ylim([0, ymax])
plt.xlim([trans_times[j][0], trans_times[j][5]])
#plt.xlim([trans_times_modif[j][1], trans_times_modif[j][6]])
lege = plt.legend(loc="upper left")
plt.vlines(x = [trans_times_modif[j][1], trans_times_modif[j][2]], ymin=0, ymax=100, colors = 'black', linestyle='dashed')
plt.hlines(y = 1, xmin=0, xmax=4e5, colors = 'black', linestyle='dashed')
plt.text(trans_times_modif[j][1] + (trans_times_modif[j][2]-trans_times_modif[j][1])*0.6, 0.6*ymax, 'ref image', horizontalalignment='center', verticalalignment='center', rotation=90)
plt.savefig(os.path.join(var.FIGS_PATH, 'history_' + canvas_parts[5].out_name(), 'attack_defense_changes_ratio.png'), bbox_inches='tight')

plt.figure()
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, deviating_pixels, label='# deviating pixels / active area')
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, pix_changes, label='# pixel changes / active area / 5 min')
plt.plot(time_ranges_stab[:-1]+time_interval_stab/2, instability, label='(1 - stability) / 1 h')
sns.despine()
plt.xlabel('Time [s]')
ymax = max(deviating_pixels[5:-1])*1.3
plt.ylim([0, ymax])
plt.xlim([trans_times[j][0], trans_times[j][5]])
#plt.xlim([trans_times_modif[j][1], trans_times_modif[j][6]])
lege = plt.legend(loc="upper left")
plt.vlines(x = [trans_times_modif[j][1], trans_times_modif[j][2]], ymin=0, ymax=100, colors = 'black', linestyle='dashed')
plt.text(trans_times_modif[j][1] + (trans_times_modif[j][2]-trans_times_modif[j][1])*0.6, 0.6*ymax, 'ref image', horizontalalignment='center', verticalalignment='center', rotation=90)
plt.savefig(os.path.join(var.FIGS_PATH, 'history_' + canvas_parts[5].out_name(), 'deviating_pixels.png'), bbox_inches='tight')

'''
# Test grid of parameters
cutoffs = np.array([0.8, 0.83, 0.85, 0.87, 0.885, 0.9, 0.915, 0.93, 0.945, 0.96])
cutoff_stables = np.array([0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 0.998])
num_stables = np.array([1, 2, 3, 4, 5, 6])
dist_stables = np.array([1, 2, 3, 4, 5])

vary_cutoffs = np.zeros((len(cutoff_stables), len(cutoffs)))
vary_dist = np.zeros((len(dist_stables), len(num_stables)))

# Vary first 2 parameters
num_stab = num_stables[3]
dist_stab = dist_stables[2]
for i in range(0, len(cutoffs)):
    print(i)
    for j in range(0, len(cutoff_stables)):
        print(j)
        vary_cutoffs[j, i] = find_all_transitions(keep_idx_comps, stability_vs_time, None, time_ranges, cutoffs[i], cutoff_stables[j], num_stab, dist_stab)[1] # num_comp_with_trans / num_comp 


# Vary last 2 parameters
cut = cutoffs[3]
cut_stab = cutoff_stables[4]
for i in range(0, len(num_stables)):
    print(i)
    for j in range(0, len(dist_stables)):
        print(j)
        vary_dist[j, i] = find_all_transitions(keep_idx_comps, stability_vs_time, None, time_ranges, cut, cut_stab, num_stables[i], dist_stables[j])[1]

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
'''