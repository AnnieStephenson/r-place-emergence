import numpy as np
import os
import canvas_part as cp
import thermo as th
import Variables.Variables as var
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

data_path = os.path.join(os.getcwd(),'data')
icp = 5 # index of the studied composition. If negative, the CanvasPart is redefined

if icp < 0: # not using the stored compositions
    x1 = 0
    x2 = 2000
    y1 = 0
    y2 = 2000

    border_path = np.array([[[x1, y1], [x1, y2], [x2, y2], [x2, y1]]])
    canvas_part = cp.CanvasPart(border_path=border_path, #id='000021', 
                                pixel_changes_all=None,
                                data_path=os.path.join(data_path, data_file='PixelChangesCondensedData_sorted.npz'))

else: # getting the composition from the stored ones
    file_path = os.path.join(data_path, 'canvas_compositions_all.pickle') 
    with open(file_path, 'rb') as f:
        canvas_parts = pickle.load(f)
    canvas_part = canvas_parts[icp]


# create directories if they don't exist yet
try:
    os.makedirs(os.path.join(os.getcwd(), 'figs', 'history_' + canvas_part.out_name()))
except OSError: 
    pass
try:
    os.makedirs(os.path.join(os.getcwd(), 'figs', 'history_' + canvas_part.out_name(), 'VsTime'))
except OSError: 
    pass


# time-dependent images + file sizes and compression
time_intervals = np.arange(0, var.TIME_TOTAL, 300)
file_size_bmp, file_size_png, t_inds_list = cp.save_part_over_time(canvas_part, time_intervals, delete_bmp=True, delete_png=False, show_plot=False)
cp.save_movie(os.path.join(os.getcwd(),'figs', 'history_' + canvas_part.out_name(),'VsTime'), 15)
cp.plot_compression(file_size_bmp, file_size_png, time_intervals, out_name = canvas_part.out_name())

# mean stability 
mean_stability = th.stability(canvas_part,np.asarray([0, var.TIME_WHITEONLY]),True,False)[0]
print('mean stability = ', mean_stability)

# time-dependent stability
time_bins = 80
time_interval = var.TIME_WHITEONLY / time_bins  # seconds
time_ranges = np.arange(0, var.TIME_WHITEONLY+time_interval-1e-4, time_interval)
stability_vs_time = th.stability(canvas_part, time_ranges, False, icp < 0)

plt.figure()
plt.plot(time_ranges[:-1]+time_interval/2, stability_vs_time)
sns.despine()
plt.ylabel('stability')
plt.xlabel('Time [s]')
plt.ylim([0, 1])
plt.xlim([0, var.TIME_TOTAL])

plt.savefig(os.path.join(os.getcwd(), 'figs', 'history_'+canvas_part.out_name(), 'stability.png'))