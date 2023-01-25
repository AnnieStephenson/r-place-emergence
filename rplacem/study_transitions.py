import numpy as np
import os
import canvas_part as cp
import thermo as th
import Variables.Variables as var
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

data_path = os.path.join(os.getcwd(),'data')
file_path_stab = os.path.join(data_path, 'stability_all_canvas_compositions.pickle')

with open(file_path_stab, 'rb') as handle:
    [mean_stability,stability_vs_time] = pickle.load(handle)

time_bins = 80
time_interval = var.TIME_WHITEONLY / time_bins  # seconds
time_ranges = np.arange(0, var.TIME_WHITEONLY+time_interval-1e-4, time_interval)

for i in range(0,30):
    print(i)
    transitions = th.find_transitions(time_ranges, stability_vs_time[i][0])
    #print(transitions[4])
    #print(transitions[5])
    #print('number of transitions = ', len(transitions[0]))
    print(transitions[0])
    print(transitions[1])