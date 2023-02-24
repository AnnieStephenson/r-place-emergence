import numpy as np
import rplacem.canvas_part as cp
import rplacem.canvas_part_statistics as stat
import rplacem.utilities as util
import rplacem.variables_rplace2022 as var
import matplotlib.pyplot as plt
import sys
import os
import glob
import json
import seaborn as sns
import pickle
import timeit

_, atlas_num = util.load_atlas()


def store_comp_stats(beg, end):
    beg = int(beg)
    end = int(end)
    file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle')
    with open(file_path, 'rb') as f:
        canvas_compositions_all = (pickle.load(f))[beg:min(end, atlas_num)]

    canvas_comp_stat_list = []
    for i in range(end-beg):
        print('CanvasComp # ' + str(beg+i) + ' , id '+ canvas_compositions_all[i].id + ' , #pixch = ', len(canvas_compositions_all[i].pixel_changes))
        canvas_comp_stat = stat.CanvasPartStatistics(canvas_compositions_all[i],
                                                    n_tbins_trans=150,
                                                    tmax=var.TIME_TOTAL,
                                                    compute_vars={'stability': 0, 
                                                                'mean_stability': 0, 
                                                                'entropy' : 0, 
                                                                'transitions' : 1, 
                                                                'attackdefense' : 0},
                                                    trans_param=[8e-3, 2e-3, 18000, 14400],
                                                    verbose=False,
                                                    renew=False,
                                                    dont_keep_dir=True)

        if canvas_comp_stat.num_transitions>0:
            print('This composition features >=1 transitions')
            compute_v = {'stability': 1, 'mean_stability': 2, 'entropy' : 2, 'transitions' : 2, 'attackdefense' : 1}
        else:
            compute_v = {'stability': 1, 'mean_stability': 2, 'entropy' : 2, 'transitions' : 0, 'attackdefense' : 0.5}

        canvas_comp_stat = stat.CanvasPartStatistics(canvas_compositions_all[i],
                                                    n_tbins=750,
                                                    n_tbins_trans=150,
                                                    tmax=var.TIME_TOTAL,
                                                    compute_vars=compute_v,
                                                    trans_param=[8e-3, 2e-3, 18000, 14400],
                                                    timeunit=300, # 5 minutes
                                                    refimage_averaging_period=3600, # 1 hour
                                                    verbose=False,
                                                    renew=True,
                                                    dont_keep_dir=True)
        
        if canvas_comp_stat.compute_vars['transitions'] == 0:                                        
            canvas_comp_stat.num_transitions == 0                               

        canvas_comp_stat_list.append(canvas_comp_stat)

    file_path = os.path.join(var.DATA_PATH, 'canvas_composition_statistics_%sto%s.pickle' %(beg, str(int(end)-1)))
    with open(file_path, 'wb') as handle:
        pickle.dump(canvas_comp_stat_list,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
        
# MAIN
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--beginning", default=0)
parser.add_argument("-e", "--end", default=1e6)
args = parser.parse_args()

store_comp_stats(args.beginning, args.end)
