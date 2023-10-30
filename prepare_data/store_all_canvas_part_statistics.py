import rplacem.canvas_part_statistics as stat
import rplacem.globalvariables_peryear as vars
var = vars.var
import os
import pickle
import gc
import math
import numpy as np
import rplacem.utilities as util

n_compositions = 12795
'''
with open(os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle'), 'rb') as f:
    print('open file')
    n_compositions = len(pickle.load(f))
    print('close file')
    f.close()
    del f
print(n_compositions)
'''

def store_comp_stats(beg, end):
    beg = int(beg)
    periodsave = 100

    end = min(int(end), n_compositions)

    for p in range(0, math.ceil((end-beg)/periodsave)):
        with open(os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle'), 'rb') as f:
            print('open file')
            canvas_compositions = (pickle.load(f))[(beg+p*periodsave):min(end, beg+(p+1)*periodsave)]
            print('close file')
            f.close()
            del f
        canvas_comp_stat_list = []

        for i in range(0, periodsave):
            if beg+p*periodsave+i >= end:
                break
            cancomp = canvas_compositions[i]
            #print('beginning:  RAM memory % used:', psutil.virtual_memory()[2],   '   ',psutil.virtual_memory()[3]/1000000000)
            print('CanvasComp # ' + str(beg+p*periodsave+i) + ' , id '+ str(cancomp.info.id) + ' , #pixch = ', len(cancomp.pixel_changes))

            #print('mid1:  RAM memory % used:', psutil.virtual_memory()[2],   '   ',psutil.virtual_memory()[3]/1000000000)
            canvas_comp_stat = stat.CanvasPartStatistics(cancomp,
                                                        t_interval=300,
                                                        tmax=var.TIME_TOTAL,
                                                        compute_vars={'stability': 1, 'entropy' : 1, 'transitions' : 1, 'attackdefense' : 1, 'other' : 1},
                                                        trans_param=[0.4, 0.15, 2*3600, 4*3600],
                                                        sliding_window=3*3600,
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

#store_comp_stats(args.beginning, args.end)


beg = np.array([i*100 for i in range(0,128)])
end_list = beg+99
end_list[-1] = n_compositions - 1
filelist = [ os.path.join(var.DATA_PATH, 'canvas_composition_statistics_%sto%s.pickle' %(b, e)) for b,e in zip(list(beg),list(end_list))]
print(filelist)
util.merge_pickles(filelist, os.path.join(var.DATA_PATH, 'canvas_composition_statistics_all.pickle' )) #canvas_composition_statistics_all.pickle
