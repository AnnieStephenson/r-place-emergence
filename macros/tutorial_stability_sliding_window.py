import numpy as np
import rplacem.canvas_part as cp
import rplacem.utilities as util
import rplacem.canvas_part_statistics as stat
import rplacem.compute_variables as compute
import rplacem.globalvariables_peryear as vars
var = vars.var
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import pickle
import matplotlib.colors as pltcolors


# Set parameters.
atlas_id_index = '000006'
n_tbins = 750
n_tbins_trans = 750
sliding_window_time = 5 * 3600
t_lims = np.linspace(0, var.TIME_TOTAL, n_tbins + 1)

# Instantiate a canvas composition object.
canvas_comp = cp.CanvasPart(id=atlas_id_index)

# Instantiate a canvas composition statistics object.

canvas_comp_stat = stat.CanvasPartStatistics(canvas_comp,
                                             n_tbins=n_tbins,
                                             compute_vars={'stability': 2,
                                                           'entropy': 2,
                                                           'transitions': 2,
                                                           'attackdefense': 2})
'''

# Use stability function to calculate the stable reference images (pixels1)
[stab_vst, _,
 pixels1,_,
 _, _, diffpix_vst,return_time] = compute.stability(canvas_comp,
                              t_lims=t_lims,
                              compute_average=True,
                              create_images=True,
                              sliding_window_time=sliding_window_time,
                              save_images=True,
                              print_progress=False)

#print(stab_vst)

# Use stability function again to get the number of differing pixels
[stab_vst2,
 _, _, _, _,_,
 diffpix_vst2,_] = compute.stability(canvas_comp,
                                      t_lims=t_lims,
                                      compute_average=True,
                                      save_images=True)

#print(stab_vst)
#print(diff_pixels_vst)


plt.plot(t_lims[1:], diffpix_vst2+500)
plt.plot(t_lims[1:], diffpix_vst)
plt.savefig(os.path.join(var.FIGS_PATH, 'test.png'))

plt.figure()
plt.plot(t_lims[1:], stab_vst2+0.1)
plt.plot(t_lims[1:], stab_vst)
plt.savefig(os.path.join(var.FIGS_PATH, 'test2.png'))

returnt_bins = np.arange(0, 3*3600 + 100, 100)
returntval = np.zeros((len(t_lims)-1, len(returnt_bins)-1))

for t in range(1, len(t_lims)):
    returntval[t-1],_ = np.histogram(return_time[t-1], bins=returnt_bins)
returntval = np.swapaxes(returntval,0,1)

tminind = np.argmax(t_lims>194000)
#for t in range(tminind,tminind+10):
    #print(t_lims[t])
    #print(return_time[t][0:1000])

fig12 = plt.figure()
plt.pcolormesh(t_lims, returnt_bins, returntval, cmap =plt.cm.jet, vmin = 0, vmax = 60)
plt.xlabel('time [sec]')
plt.ylabel('return time')
plt.colorbar(label='number of pixels')
#plt.xlim([170000,230000])
#plt.ylim([0.9, 800])
plt.savefig(os.path.join(var.FIGS_PATH,'test_returntime'), bbox_inches='tight')

returntmed = np.empty((len(t_lims)-1))
for t in range(1, len(t_lims)):
    returntmed[t-1] = np.median(return_time[t-1][return_time[t-1]>0])

plt.figure()
plt.plot(t_lims[1:], returntmed)
plt.savefig(os.path.join(var.FIGS_PATH, 'test_returntime2.png'))

'''