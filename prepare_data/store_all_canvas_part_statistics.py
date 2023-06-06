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
import psutil
import gc
import cProfile

_, atlas_num = util.load_atlas()


def store_comp_stats(beg, end):
    beg = int(beg)
    end = min(int(end), atlas_num)
    periodsave = 100

    for p in range(0, int((end-beg)/periodsave)):
        with open(os.path.join(var.DATA_PATH, 'canvas_compositions_files0to99.pickle'), 'rb') as f:
            print('open file')
            canvas_compositions = (pickle.load(f))[(beg+p*periodsave):min(end, beg+(p+1)*periodsave)]
            print('close file')
            f.close()
            del f
        canvas_comp_stat_list = []

        for i in range(0, periodsave):
            if beg+p*periodsave+i > end:
                break
            cancomp = canvas_compositions[i]
            #print('beginning:  RAM memory % used:', psutil.virtual_memory()[2],   '   ',psutil.virtual_memory()[3]/1000000000)
            print('CanvasComp # ' + str(beg+p*periodsave+i) + ' , id '+ str(cancomp.info.id) + ' , #pixch = ', len(cancomp.pixel_changes))
            
            '''
            canvas_comp_stat = stat.CanvasPartStatistics(cancomp,
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
                compute_v = {'stability': 1, 'mean_stability': 2, 'entropy' : 1, 'transitions' : 2, 'attackdefense' : 1}
            else:
                compute_v = {'stability': 1, 'mean_stability': 2, 'entropy' : 1, 'transitions' : 0, 'attackdefense' : 0.5}
            '''

            if cancomp.has_loc_jump: #should fix that in canvas compositions...
                continue
            #print('mid1:  RAM memory % used:', psutil.virtual_memory()[2],   '   ',psutil.virtual_memory()[3]/1000000000)
            canvas_comp_stat = stat.CanvasPartStatistics(cancomp,
                                                        t_interval=300,
                                                        tmax=var.TIME_TOTAL,
                                                        compute_vars={'stability': 1, 'entropy' : 1, 'transitions' : 1, 'attackdefense' : 1, 'other' : 1},
                                                        trans_param=[0.4, 0.15, 2*3600, 4*3600],
                                                        sliding_window=14400,
                                                        timeunit=300, # 5 minutes
                                                        verbose=False,
                                                        renew=True,
                                                        dont_keep_dir=True,
                                                        flattening='ravel',#'hilbert_pkg',
                                                        compression='DEFLATE',#'LZ77'
                                                        )

            #print('mid:  RAM memory % used:', psutil.virtual_memory()[2],   '   ',psutil.virtual_memory()[3]/1000000000)
            canvas_comp_stat_list.append(canvas_comp_stat)
            del canvas_comp_stat

            gc.collect()


        print('save')
        file_path = os.path.join(var.DATA_PATH, 'canvas_composition_statistics_%sto%s.pickle' %(str(beg+p*periodsave), str(min(end, beg+(p+1)*periodsave) - 1))) #str(int(end)-1)))
        with open(file_path, 'wb') as handle:
            pickle.dump(canvas_comp_stat_list,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        del canvas_comp_stat_list
        del canvas_compositions
        gc.collect()

        #print('end:  RAM memory % used:', psutil.virtual_memory()[2],   '   ',psutil.virtual_memory()[3]/1000000000)
        
    #with open(os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle'), 'rb') as f:
    #    print(pickle.load(f))

# MAIN
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--beginning", default=0)
parser.add_argument("-e", "--end", default=1e6)
args = parser.parse_args()

store_comp_stats(args.beginning, args.end)

'''
beg = np.array([i*100 for i in range(0,2)]) #until 9900
end = beg+99
end[-1] = atlas_num - 1
filelist = [ os.path.join(var.DATA_PATH, 'canvas_composition_statistics_%sto%s.pickle' %(b, e)) for b,e in zip(list(beg),list(end))]
print(filelist)
util.merge_pickles(filelist, os.path.join(var.DATA_PATH, 'canvas_composition_statistics_all.pickle' )) #canvas_composition_statistics_all.pickle
'''