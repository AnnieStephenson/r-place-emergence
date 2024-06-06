import os
import numpy as np
import rplacem.canvas_part as cp
import json
import pickle
import rplacem.globalvariables_peryear as vars
var = vars.var
import rplacem.utilities as util

atlas, atlas_num = util.load_atlas()

def store_compos(beg, end):
    file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_files%sto%s.pickle' %(beg,str(int(end)-1)))
    canvas_comps = []

    pixel_changes_all = util.get_all_pixel_changes()

    print('start looping over compositions')
    for i in range(int(beg),int(end)):
        print(i, atlas[i]['id'])
        atlas_info_separated = cp.get_atlas_border(id_index=i, atlas=atlas, addtime_before=10*3600, addtime_after=5*3600) #used 7h before and after in the previous run
        for ainfo in atlas_info_separated:
            # actual canvas composition here
            #print(ainfo.id, ainfo.border_path, ainfo.border_path_times)
            canvas_comps.append( cp.CanvasPart(atlas_info=ainfo, pixel_changes_all=pixel_changes_all) )

    with open(file_path, 'wb') as handle:
        pickle.dump(canvas_comps,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

# MAIN
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--beginning", default=0)
parser.add_argument("-e", "--end", default=atlas_num)
args = parser.parse_args()

print(atlas_num)
#store_compos(args.beginning, args.end)

'''
util.merge_pickles([ 
                'data/'+str(var.year)+'/canvas_compositions_files0to299.pickle',
                'data/'+str(var.year)+'/canvas_compositions_files300to599.pickle',
                'data/'+str(var.year)+'/canvas_compositions_files600to1599.pickle',
                'data/'+str(var.year)+'/canvas_compositions_files1600to2599.pickle',
                'data/'+str(var.year)+'/canvas_compositions_files2600to3099.pickle',
                'data/'+str(var.year)+'/canvas_compositions_files3100to3599.pickle',
                'data/'+str(var.year)+'/canvas_compositions_files3600to4599.pickle',
                'data/'+str(var.year)+'/canvas_compositions_files4600to6422.pickle',
                #'data/'+str(var.year)+'/canvas_compositions_files10200to10884.pickle',
                ],
                'canvas_compositions_all.pickle')
'''

cp.clean_all_compositions(os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle'),
                          os.path.join(var.DATA_PATH, 'canvas_compositions_all_cleaned.pickle'))