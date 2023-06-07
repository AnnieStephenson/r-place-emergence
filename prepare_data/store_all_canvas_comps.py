import os
import numpy as np
import rplacem.canvas_part as cp
import json
import pickle
import rplacem.variables_rplace2022 as var
import rplacem.utilities as util

atlas, atlas_num = util.load_atlas()

def store_compos(beg, end):
    file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_files%sto%s.pickle' %(beg,str(int(end)-1)))
    canvas_comps = []

    pixel_changes_all = util.get_all_pixel_changes()

    print('start looping over compositions')
    for i in range(int(beg),int(end)):
        print(i, atlas[i]['id'])
        atlas_info_separated = cp.get_atlas_border(id_index=i, atlas=atlas, addtime_before=7*3600, addtime_after=7*3600)
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

store_compos(args.beginning, args.end)

'''
util.merge_pickles([ 'data/canvas_compositions_files0to999.pickle',
                'data/canvas_compositions_files1000to1999.pickle',
                'data/canvas_compositions_files2000to2999.pickle',
                'data/canvas_compositions_files3000to3999.pickle',
                'data/canvas_compositions_files4000to4999.pickle',
                'data/canvas_compositions_files5000to6999.pickle',
                'data/canvas_compositions_files7000to8999.pickle',
                'data/canvas_compositions_files9000to10630.pickle',
                ],
                'canvas_compositions_all.pickle')
'''