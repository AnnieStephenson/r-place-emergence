import numpy as np
import os
import rplacem.canvas_part as cp
import rplacem.canvas_part_statistics as stat
import rplacem.compute_variables as comp
import rplacem.utilities as util
import rplacem.variables_rplace2022 as var
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

'''
icp = 5 # index of the studied composition. If negative, the CanvasPart is redefined

if icp < 0: # not using the stored compositions
    x1 = 0
    x2 = 1999
    y1 = 0
    y2 = 1999

    border_path = np.array([[[x1, y1], [x1, y2], [x2, y2], [x2, y1]]])
    canpart = cp.CanvasPart(border_path=border_path, #id='000021', 
                                pixel_changes_all=None) 

else: # getting the composition from the stored ones
    file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle') 
    with open(file_path, 'rb') as f:
        canvas_parts = pickle.load(f)
    canpart = canvas_parts[icp]
    print(canpart)

'''

# only quick check here
pixel_changes_all = util.get_all_pixel_changes()
atlas, _ = util.load_atlas()

canvas_comps = []
file_path = os.path.join(var.DATA_PATH, 'CanvasParts_test.pickle')
with open(file_path, 'rb') as f:
    canvas_comps = pickle.load(f)

'''
n = 300
isrect = np.zeros(n)
npix = np.zeros(n)
bmpsize_over_npix = np.zeros(n)
bmpsize_over_npix_rectequiv = np.zeros(n)

for i in range(0,n):
    print(i)
    print(atlas[i]['id'])
    print(atlas[i]['links'])
    #canvas_comps.append( cp.CanvasPart(id=atlas[i]['id'], pixel_changes_all=pixel_changes_all, atlas=atlas) )
    isrect[i] = canvas_comps[i].is_rectangle
    npix[i] = len(canvas_comps[i].coords[0])
    cpstat = stat.CanvasPartStatistics(canvas_comps[i], verbose=True, compute_vars={'stability': 1, 'mean_stability': 1, 'entropy' : 1}, dont_keep_dir=True)
    bmpsize_over_npix[i] = cpstat.bmpsize / cpstat.area
    bmpsize_over_npix_rectequiv[i] = cpstat.bmpsize / cpstat.area_rectangle


#with open(file_path, 'wb') as handle:
#    pickle.dump(canvas_comps,
#                handle,
#                protocol=pickle.HIGHEST_PROTOCOL)


# scatter plot
fig = plt.figure()
plt.scatter(npix, bmpsize_over_npix,  s=3)
plt.ylabel('bmp file size / # active pixels')
plt.xlabel('# active pixels')
plt.xscale('log')
plt.xlim(1,max(npix)*1.3)
plt.savefig(os.path.join(var.FIGS_PATH, 'bmpfilesize_over_npix.png'), bbox_inches='tight')

# scatter plot
fig = plt.figure()
plt.scatter(npix, bmpsize_over_npix_rectequiv,  s=3)
plt.ylabel('bmp file size / # active pixels')
plt.xlabel('# active pixels')
plt.xscale('log')
plt.xlim(1,max(npix)*1.3)
plt.savefig(os.path.join(var.FIGS_PATH, 'bmpfilesize_over_npix__rectangle_equivalent.png'), bbox_inches='tight')

# scatter plot
fig2 = plt.figure()
plt.scatter(npix[np.where(isrect)[0]], bmpsize_over_npix[np.where(isrect)[0]], s=3)
plt.ylabel('bmp file size / # active pixels')
plt.xlabel('# active pixels')
plt.xscale('log')
plt.xlim(1,max(npix)*1.3)
plt.savefig(os.path.join(var.FIGS_PATH, 'bmpfilesize_over_npix_onlyrectangles.png'), bbox_inches='tight')
'''

'''
canpart = cp.CanvasPart(
                        #border_path=[[[0, 0], [0, 100], [100, 100], [100, 0]]],
                        id='000006', 
                        pixel_changes_all=pixel_changes_all,
                        verbose=True, save=True)
'''

ntrans = 0
for i in range(0,200):
    canpart = canvas_comps[i]
    print(i, canpart.id)
    cpstat = stat.CanvasPartStatistics(canpart, n_tbins=100, 
                                    compute_vars={'stability': 1, 'mean_stability': 1, 'entropy' : 1, 'transitions' : 1, 'attackdefense' : 1},
                                    verbose=False, dont_keep_dir=False)
    print(cpstat.transition_times)
    if len(cpstat.transition_times)>0:
        ntrans += 1

print('frac of compos with a trans = ',ntrans/200)

'''
plt.figure()
plt.plot(cpstat.t_ranges[1:] - cpstat.t_interval / 2, cpstat.instability_vst_norm, label='instability ')
plt.plot(cpstat100.t_ranges[1:] - cpstat100.t_interval / 2, cpstat100.instability_vst_norm, label='instability (100 time bins)')
plt.plot(cpstat200.t_ranges[1:] - cpstat200.t_interval / 2, cpstat200.instability_vst_norm, label='instability (200 time bins)')
plt.plot(cpstat1000.t_ranges[1:] - cpstat1000.t_interval / 2, cpstat1000.instability_vst_norm, label='instability (1000 time bins)')
sns.despine()
plt.legend(loc="upper left")
plt.ylabel('instability / 5 min')
plt.xlabel('Time [s]')
plt.ylim([0.0001, max(cpstat1000.instability_vst_norm)*0.2])
plt.xlim([120000, 250000])
plt.yscale('log')
plt.savefig(os.path.join(var.FIGS_PATH, canpart.out_name(), 'instability_various_time_intervals.png'), bbox_inches='tight')
'''


'''
# create directories if they don't exist yet
try:
    os.makedirs(os.path.join(var.FIGS_PATH, canpart.out_name()))
except OSError: 
    pass
try:
    os.makedirs(os.path.join(var.FIGS_PATH, canpart.out_name(), 'VsTime'))
except OSError: 
    pass

# time-dependent images + file sizes and compression
time_intervals = np.arange(0, var.TIME_TOTAL, 300)
file_size_bmp, file_size_png, t_inds_list = comp.save_part_over_time(canpart, time_intervals, delete_bmp=True, delete_png=False, show_plot=False)
util.save_movie(os.path.join(var.FIGS_PATH, canpart.out_name(),'VsTime'), 15)
comp.plot_compression(file_size_bmp, file_size_png, time_intervals, out_name = canpart.out_name())

# mean stability 
mean_stability = comp.stability(canpart,np.asarray([0, var.TIME_WHITEONLY]), True,True,False,True)[0][0]
print('mean stability = ', mean_stability)

# time-dependent stability
time_bins = 80
time_interval = var.TIME_WHITEONLY / time_bins  # seconds
time_ranges = np.arange(0, var.TIME_WHITEONLY+time_interval-1e-4, time_interval)
stability_vs_time = comp.stability(canpart, time_ranges, False, False, 
                                   False, #icp < 0, 
                                   True)[0]

plt.figure()
plt.plot(time_ranges[:-1]+time_interval/2, stability_vs_time)
sns.despine()
plt.ylabel('stability')
plt.xlabel('Time [s]')
plt.ylim([0, 1])
plt.xlim([0, var.TIME_TOTAL])

plt.savefig(os.path.join(var.FIGS_PATH, canpart.out_name(), 'stability.png'))

transitions = comp.find_transitions(time_ranges, stability_vs_time)
print(transitions)
print(canpart.border_path_times)

'''