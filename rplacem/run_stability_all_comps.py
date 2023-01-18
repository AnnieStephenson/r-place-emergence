import numpy as np
import os
import canvas_part as cp
import thermo as th
import Variables.Variables as var
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

data_path = os.path.join(os.getcwd(),'data')
file_path = os.path.join(data_path, 'canvas_compositions_all.pickle') 

# grab all compositions
with open(file_path, 'rb') as f:
    canvas_parts = pickle.load(f)

# time intervals
time_bins = 80
time_interval = var.TIME_WHITEONLY / time_bins  # seconds
time_ranges = np.arange(0, var.TIME_WHITEONLY+time_interval-1e-4, time_interval)

mean_stability = []
stability_vs_time = []
file_path_stab = os.path.join(data_path, 'stability_all_canvas_compositions.pickle')

# loop over compositions
for i in range(0, len(canvas_parts)):
    if i%100 == 0:
        print(i)
    # mean stability
    mean_stability.append( th.stability(canvas_parts[i],np.asarray([0, var.TIME_WHITEONLY]),False,False)[0] )

    # time dependent stability
    stability_vs_time.append( [th.stability(canvas_parts[i], time_ranges, False,False)] )
    print(mean_stability[i])

# store stability for all compositions
with open(file_path_stab, 'wb') as handle:
    pickle.dump([time_ranges, mean_stability, stability_vs_time],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)

# plot mean stability for all compositions
with open(file_path_stab, 'rb') as handle:
    [time_bins, mean_stability, stability_vs_time] = pickle.load(handle)

n, bins, patches = plt.hist(mean_stability, 70, range=[0,1], facecolor='blue', alpha=0.5, align='mid')

sns.despine()
plt.xlabel('mean stability')
plt.ylabel('number of compositions')
plt.xlim([0, 1])
plt.savefig(os.path.join(os.getcwd(), 'figs','mean_stability_for_all_compositions.png'))

# plot time-dependent stability for all compositions
plt.figure()
for i in range(0,len(canvas_parts)):
    plt.plot(time_bins[:-1] + (time_bins[1]-time_bins[0]) / 2, stability_vs_time[i][0])

sns.despine()
plt.ylabel('stability')
plt.xlabel('Time [s]')
plt.ylim([0, 1])
plt.xlim([0, var.TIME_TOTAL])

plt.savefig(os.path.join(os.getcwd(), 'figs', 'stability_vs_time_all_compositions.png'))