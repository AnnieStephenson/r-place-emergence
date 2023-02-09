import numpy as np
import os
import numpy as np
import rplacem.canvas_part as cp
import json
import pickle
import rplacem.utilities as util
import cProfile, pstats, io

data_path = os.path.join(os.getcwd(),'data')
composition_classification_file = open( os.path.join(data_path,'atlas.json') )
atlas = json.load(composition_classification_file)
atlas_num = len(atlas)

def store_compos(beg, end):
    file_path = os.path.join(data_path, 'canvas_compositions_files%sto%s.pickle' %(beg,str(int(end)-1)))
    canvas_comps = []

    pixel_changes_all = util.get_all_pixel_changes()

    print('start looping over compositions')

    #from pstats import SortKey
    #pr = cProfile.Profile()
    #pr.enable()

    for i in range(int(beg),int(end)):
        print(i)
        print(atlas[i]['id'])
        canvas_comps.append( cp.CanvasPart(id=atlas[i]['id'], pixel_changes_all=pixel_changes_all, atlas=atlas) )

    #pr.disable()
    #s = io.StringIO()
    #sortby = SortKey.CUMULATIVE
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())

    with open(file_path, 'wb') as handle:
        pickle.dump(canvas_comps,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

def merge_pickles(file_list):
    out = []
    for file in file_list:
        with open(file, 'rb') as f:
            out += pickle.load(f)
    
    with open(os.path.join(data_path, 'canvas_compositions_all.pickle'), 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

# MAIN
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--beginning", default=0)
parser.add_argument("-e", "--end", default=atlas_num)
args = parser.parse_args()

#store_compos(args.beginning, args.end)

merge_pickles([ 'data/canvas_compositions_files0to2199.pickle',
                'data/canvas_compositions_files2200to4399.pickle',
                'data/canvas_compositions_files4400to6599.pickle',
                'data/canvas_compositions_files6600to7799.pickle',
                'data/canvas_compositions_files7800to9199.pickle',
                'data/canvas_compositions_files9200to9317.pickle',
                ])