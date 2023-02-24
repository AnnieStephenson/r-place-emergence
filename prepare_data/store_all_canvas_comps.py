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
        canvas_comps.append( cp.CanvasPart(id=atlas[i]['id'], pixel_changes_all=pixel_changes_all, atlas=atlas) )

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

util.merge_pickles([ 'data/canvas_compositions_files0to1299.pickle',
                'data/canvas_compositions_files1300to2599.pickle',
                'data/canvas_compositions_files2600to3899.pickle',
                'data/canvas_compositions_files3900to5199.pickle',
                'data/canvas_compositions_files5200to6499.pickle',
                'data/canvas_compositions_files6500to7199.pickle',
                'data/canvas_compositions_files7200to8399.pickle',
                'data/canvas_compositions_files8400to9957.pickle',
                ],
                'canvas_compositions_all.pickle')
