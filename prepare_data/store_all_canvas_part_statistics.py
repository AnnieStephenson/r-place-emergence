import rplacem.canvas_part_statistics as stat
import rplacem.globalvariables_peryear as vars
var = vars.var
import os
import pickle
import gc
import math
import numpy as np
import rplacem.utilities as util

file_ext = ['all'] if var.year == 2022 else \
           ['files0to299', 'files300to599', 'files600to1599', 'files1600to2599', 'files2600to3099', 'files3100to3599', 'files3600to4599', 'files4600to6422']
n_compos = np.array([360, 339, 1060, 1026, 513, 511, 1033, 1878]) if var.year == 2023 else np.array([14222])
n_comp_tot = np.cumsum(n_compos)

'''
for ext in file_ext:
    with open(os.path.join(var.DATA_PATH, 'canvas_compositions_'+ext+'.pickle'), 'rb') as f:
        n_compos.append(len(pickle.load(f)))
        f.close()
        del f
print(n_compos)
'''

def store_comp_stats(beg, end):
    beg = int(beg)
    periodsave = 100

    end = min(int(end), np.sum(n_compos))

    for p in range(0, math.ceil((end-beg)/periodsave)):
        ifile = np.argmax(beg+p*periodsave <= n_comp_tot - 1)
        with open(os.path.join(var.DATA_PATH, 'canvas_compositions_'+file_ext[ifile]+'.pickle'), 'rb') as f:
            print('open file')
            shifted_start_idx = beg+p*periodsave-(n_comp_tot[ifile-1] if ifile>0 else 0)
            canvas_compositions = (pickle.load(f))[shifted_start_idx:min(n_comp_tot[ifile], shifted_start_idx + periodsave)]
            print('close file')
            f.close()
            del f
        canvas_comp_stat_list = []

        for i in range(0, periodsave):
            if beg+p*periodsave+i >= end:
                break

            if beg+p*periodsave+i >= n_comp_tot[ifile]:
                ifile += 1
                with open(os.path.join(var.DATA_PATH, 'canvas_compositions_'+file_ext[ifile]+'.pickle'), 'rb') as f:
                    print('open file')
                    canvas_compositions += (pickle.load(f))[0:min(n_comp_tot[min(ifile+1, len(n_comp_tot)-1)], beg-n_comp_tot[ifile]+(p+1)*periodsave)]
                    print('close file')
                    f.close()
                    del f

            cancomp = canvas_compositions[i]
            if cancomp.info.id == '4839' or cancomp.info.id == '3252':
                continue
            #print('beginning:  RAM memory % used:', psutil.virtual_memory()[2],   '   ',psutil.virtual_memory()[3]/1000000000)
            print('CanvasComp # ' + str(beg+p*periodsave+i) + ' , id '+ str(cancomp.info.id) + ' , #pixch = ', len(cancomp.pixel_changes))
            if len(cancomp.pixel_changes) > 50000:
                continue

            #print('mid1:  RAM memory % used:', psutil.virtual_memory()[2],   '   ',psutil.virtual_memory()[3]/1000000000)
            canvas_comp_stat = stat.CanvasPartStatistics(cancomp,
                                                        t_interval=300,
                                                        tmax=var.TIME_TOTAL,
                                                        compute_vars={'stability': 1, 'entropy' : 1, 'transitions' : 1, 'attackdefense' : 1, 'other' : 1, 'ews' : 1},
                                                        trans_param=[0.35, 6],#, 2*3600, 4*3600],
                                                        sliding_window=int(3*3600),
                                                        timeunit=300, # 5 minutes
                                                        verbose=False,
                                                        renew=True,
                                                        dont_keep_dir=True,
                                                        flattening='ravel',#'hilbert_pkg',
                                                        compression='LZ77'
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
        
# MAIN
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--beginning", default=0)
parser.add_argument("-e", "--end", default=1e6)
args = parser.parse_args()

store_comp_stats(args.beginning, args.end)


beg = np.array([i*100 for i in range(0,int(n_comp_tot[-1]/100)+1)])
end_list = beg+99
end_list[-1] = n_comp_tot[-1] - 1
filelist = [ os.path.join(var.DATA_PATH, 'canvas_composition_statistics_%sto%s.pickle' %(b, e)) for b,e in zip(list(beg),list(end_list))]
#print(filelist)
#util.merge_pickles(filelist, os.path.join(var.DATA_PATH, 'canvas_composition_statistics_all.pickle' ))
