import rplacem.plot_utilities as plot
import rplacem.globalvariables_peryear as vars
var = vars.var
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

ids = []
times = []
for cps in cpstats:
    if cps.n_transitions > 0 and 'part' in cps.id:
        ids.append(cps.id)
        times.append(cps.transition_times)

ids2 = []
times2 = []
for i in range(0,len(times)):
    if i==0:
        if ids[i][0:6] == ids[i+1][0:6]:
            ids2.append(ids[i])
            times2.append(times[i])
    if ids[i][0:6] == ids[i-1][0:6] or ids[i][0:6] == ids[i+1][0:6]:
        ids2.append(ids[i])
        times2.append(times[i])

count=0
count2=0
for i in range(1,len(times2)):
    if ids2[i][0:6] == ids2[i-1][0:6] and (abs(times2[i][0][0]-times2[i-1][-1][0])<3000 or abs(times2[i][-1][0]-times2[i-1][0][0])<3000):
        #print(ids2[i],ids2[i-1],times2[i], times2[i-1], times2[i][0][0]-times2[i-1][-1][0])
        count += 1


pbm_ids = []
for i in range(1,len(times2)):
    if ids2[i][0:6] == ids2[i-1][0:6] and (abs(times2[i][0][0]-times2[i-1][-1][0])<301 or abs(times2[i][-1][0]-times2[i-1][0][0])<301):
        pbm_ids.append([ids2[i-1], ids2[i]])
        count2+=1

for id1,id2 in pbm_ids:
    cps1,cps2=None,None
    for cps in cpstats:
        if cps.id==id1:
            cps1 = cps
        if cps.id==id2:
            cps2 = cps
    # check for spatial overlap

print(count)
print(count2)

