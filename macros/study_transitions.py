import numpy as np
import os
import rplacem.canvas_part as cp
import rplacem.compute_variables as comp
import rplacem.variables_rplace2022 as var
import rplacem.transitions as tr
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import rplacem.compute_variables as comp
import rplacem.variables_rplace2022 as var

# get time-dependent stability for all compos
file_path_stab = os.path.join(var.DATA_PATH, 'stability_all_canvas_compositions.pickle')

with open(file_path_stab, 'rb') as handle:
    [mean_stability,stability_vs_time] = pickle.load(handle)

# get all compos
file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle') 
with open(file_path, 'rb') as f:
    canparts = pickle.load(f)

canpart = cp.CanvasPart(id='000006', pixel_changes_all=None)
print(canpart.coords)
# test only compositions of significant size

time_bins_stab = 80
time_bins_trans = 50
time_interval_stab = var.TIME_WHITEONLY / time_bins_stab  # seconds
time_interval_trans = var.TIME_WHITEONLY / time_bins_trans  # seconds
time_ranges_stab = np.arange(0, var.TIME_WHITEONLY+time_interval_stab-1e-4, time_interval_stab)
time_ranges_trans = np.arange(0, var.TIME_WHITEONLY+time_interval_trans-1e-4, time_interval_trans)

j = 0
#tr.find_all_transitions(keep_idx_comps, stability_vs_time, time_ranges_stab, canparts, 0.88, 0.985, 3, 3, True)
image_pre, _, _, trans_times, trans_times_modif = tr.transition_reference_image(canpart, time_ranges_stab, stability_vs_time[5], True, 0.88, 0.99, 3, 4)
res = comp.num_changes_and_users(canpart, time_ranges_trans, image_pre[j], True)
pix_changes = res[0] * 300 / (time_interval_trans * res[3]) 
defense_changes = res[1] * 300 / (time_interval_trans * res[3]) 
attack_changes = res[2] * 300 / (time_interval_trans * res[3])
deviating_pixels = res[4] / res[3]
instability = (1-stability_vs_time[5][0]) * 3600 / time_interval_stab
num_attackonly_users = res[5] * 3600 / (time_interval_trans * res[8])
num_defenseonly_users = res[6] * 3600 / (time_interval_trans * res[8])
num_attackdefense_users = res[7] * 3600 / (time_interval_trans * res[8])

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
plt.savefig(os.path.join(var.FIGS_PATH, canpart.out_name(), 'number_attack_pixel_changes_timeinterval{:.0f}s.png'.format(time_interval_trans)), bbox_inches='tight')


with np.errstate(divide='ignore', invalid='ignore'):
    attack_defense_ratio = attack_changes / defense_changes
attack_defense_ratio[np.where(defense_changes == 0)] = 0.
plt.figure()
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, attack_defense_ratio, label='# attack / # defense changes')
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, pix_changes, label='# pixel changes / active area / 5 min')
plt.plot(time_ranges_stab[:-1]+time_interval_stab/2, instability, label='(1 - stability) / 1 h')
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, deviating_pixels, label='# deviating pixels / active area')
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
plt.savefig(os.path.join(var.FIGS_PATH, canpart.out_name(), 'attack_defense_changes_ratio_timeinterval{:.0f}s.png'.format(time_interval_trans)), bbox_inches='tight')

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
plt.savefig(os.path.join(var.FIGS_PATH, canpart.out_name(), 'deviating_pixels_timeinterval{:.0f}s.png'.format(time_interval_trans)), bbox_inches='tight')

with np.errstate(divide='ignore', invalid='ignore'):
    onlyattackordefense_users_ratio = num_attackonly_users / num_defenseonly_users
attackordefense_users_ratio = (num_attackonly_users + num_attackdefense_users) / (num_defenseonly_users + num_attackdefense_users)
plt.figure()
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, onlyattackordefense_users_ratio, label='# only attack / # only defend users')
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, attackordefense_users_ratio, label='# attack / # defend users')
plt.plot(time_ranges_stab[:-1]+time_interval_stab/2, instability, label='(1 - stability) / 1 h')
#plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, num_attackonly_users, label='# users that only attack')
#plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, num_defenseonly_users, label='# users that only defend')
#plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, num_attackdefense_users, label='# users doing both')
sns.despine()
plt.xlabel('Time [s]')
ymax = max(onlyattackordefense_users_ratio[5:-1])*1.2
plt.ylim([0, ymax])
plt.xlim([trans_times[j][0], trans_times[j][5]])
#plt.xlim([trans_times_modif[j][1], trans_times_modif[j][6]])
lege = plt.legend(loc="upper left")
plt.hlines(y = 1, xmin=0, xmax=4e5, colors = 'black', linestyle='dashed')
plt.vlines(x = [trans_times_modif[j][1], trans_times_modif[j][2]], ymin=0, ymax=ymax, colors = 'black', linestyle='dashed')
plt.text(trans_times_modif[j][1] + (trans_times_modif[j][2]-trans_times_modif[j][1])*0.6, 0.6*ymax, 'ref image', horizontalalignment='center', verticalalignment='center', rotation=90)
plt.savefig(os.path.join(var.FIGS_PATH, canpart.out_name(), 'attacking_vs_defending_users_ratio_timeinterval{:.0f}s.png'.format(time_interval_trans)), bbox_inches='tight')

plt.figure()
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, num_attackonly_users, label='# users that only attack / # total users / 1h')
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, num_defenseonly_users, label='# users that only defend / # total users / 1h')
plt.plot(time_ranges_trans[:-1]+time_interval_trans/2, num_attackdefense_users, label='# users doing both / # total users / 1h')
plt.plot(time_ranges_stab[:-1]+time_interval_stab/2, instability, label='(1 - stability) / 1 h')
sns.despine()
plt.xlabel('Time [s]')
ymax = 0.10#max(num_attackonly_users[5:-1])*1.4
plt.ylim([0, ymax])
plt.xlim([trans_times[j][0], trans_times[j][5]])
#plt.xlim([trans_times_modif[j][1], trans_times_modif[j][6]])
lege = plt.legend(loc="upper left")
plt.vlines(x = [trans_times_modif[j][1], trans_times_modif[j][2]], ymin=0, ymax=ymax, colors = 'black', linestyle='dashed')
plt.text(trans_times_modif[j][1] + (trans_times_modif[j][2]-trans_times_modif[j][1])*0.6, 0.6*ymax, 'ref image', horizontalalignment='center', verticalalignment='center', rotation=90)
plt.savefig(os.path.join(var.FIGS_PATH, canpart.out_name(), 'attacking_vs_defending_users_timeinterval{:.0f}s.png'.format(time_interval_trans)), bbox_inches='tight')