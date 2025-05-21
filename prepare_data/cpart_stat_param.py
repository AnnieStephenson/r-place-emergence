import numpy as np
#year = 2022
#rplacem.set_year(year)
from rplacem import var as var
import rplacem.utilities as util
import rplacem.canvas_part_statistics as cpst
import os
import pickle

comp_path = os.path.join(var.DATA_PATH, 'canvas_comps_'+ str(var.year) + '_clean.pkl')
with open(comp_path, 'rb') as f:
    canvas_comps = pickle.load(f)

###### For SLURM ###########################

cpart_chunks = [[0, 1200], 
                [1200, 3500], 
                [3500, 6000],
                [6000, 9000], 
                [9000, len(canvas_comps)]]
n_params = len(var.sw_str)
tot_jobs = len(var.sw_str)*len(cpart_chunks)

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
sw_param = var.sw_str[idx % n_params]
trans_abs_param = var.ta_str[idx % n_params]
trans_rel_param = var.tr_str[idx % n_params]
cpart_inds = cpart_chunks[int(np.floor(idx / n_params))] 
############################################
print(str(var.year), flush=True)
atlas, atlas_size = util.load_atlas()
pixel_changes_all = util.get_all_pixel_changes()
cpart_stats = []
for i in range(cpart_inds[0], cpart_inds[1]):
    if i > len(canvas_comps)-1:
        break
    print('i: ' + str(i),flush=True)
    print('Atlas name: ' + str(canvas_comps[i].info.atlasname), flush=True) 
    print('\n', flush=True)
    canvas_part_stat = cpst.CanvasPartStatistics(canvas_comps[i],
                                                    t_interval=300,
                                                    tmax=var.TIME_WHITEOUT,
                                                    compute_vars={'stability': 1, 
                                                                'entropy': 1, 
                                                                'inout':0,
                                                                'transitions':1, 
                                                                'attackdefense': 1, 
                                                                'lifetime_vars' : 0,
                                                                'other': 1,
                                                                'ews':1,
                                                                'void_attack':0},
                                                    sliding_window=sw_param*3600,
                                                    trans_param=[trans_abs_param, trans_rel_param],
                                                    compression='LZ77',
                                                    flattening='ravel')
   
    cpart_stats.append(canvas_part_stat)

cstat_filename = 'cpart_stats_sw' + str(sw_param) + '_ta' +str(trans_abs_param) + '_tr' + str(trans_rel_param)  + '_cp' + str(cpart_inds[0]) + '-' + str(cpart_inds[1]) + '_' + str(var.year) + '.pkl'
with open(os.path.join(var.DATA_PATH, cstat_filename), 'wb') as file:
    pickle.dump(cpart_stats, file)
print(f"INFO: Job with array task id {idx} is using myparam={sw_param}")
with open(f"output_taskid_{idx}_myparam_{sw_param}.out", "w") as f:
    msg = f"Output file for task id {idx} using myparam={sw_param}\n"
    f.write(msg)
