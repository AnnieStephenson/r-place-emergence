import rplacem.plot_utilities as plot
import rplacem.variables_rplace2022 as var
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

file_path = os.path.join(var.DATA_PATH, 'canvas_composition_statistics_all.pickle') 
with open(file_path, 'rb') as f:
    cpstats = pickle.load(f)

n_trans = 0
n_comps = 0
n_comp_with_trans = 0

for cps in cpstats:
    if cps.area >= 50:
        n_comps += 1
        n_trans += cps.n_transitions
        if cps.n_transitions > 0:
            n_comp_with_trans += 1

print(n_trans,'transitions from',n_comps,'compositions of > 50 pixels')
print(n_comp_with_trans,'compositions show at least 1 transition')
