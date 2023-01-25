import numpy as np
import os
import cProfile, pstats, io
from Variables import * 
import numpy as np
import rplacem.canvas_part as cp
import rplacem.thermo as th
import matplotlib.pyplot as plt
import sys
import glob
import json
import seaborn as sns
import pickle

data_path = os.path.join(os.getcwd(),'data')
composition_classification_file = open( os.path.join(data_path,'atlas.json') )
atlas = json.load(composition_classification_file)
atlas_num = len(atlas)

def store_compos(beg, end):
    file_path = os.path.join(data_path, 'canvas_compositions_files%sto%s.pickle' %(beg,str(int(end)-1)))
    canvas_comps = []

    pixel_changes_all_npz = np.load(os.path.join(data_path, 'PixelChangesCondensedData_sorted.npz'))
                                                
    pixel_changes_all = np.core.records.fromarrays( [np.array(pixel_changes_all_npz['seconds'], dtype=np.float64),
                                                    np.array(pixel_changes_all_npz['pixelXpos'], dtype=np.uint16),
                                                    np.array(pixel_changes_all_npz['pixelYpos'], dtype=np.uint16),
                                                    np.array(pixel_changes_all_npz['userIndex'], dtype=np.uint32),
                                                    np.array(pixel_changes_all_npz['colorIndex'], dtype=np.uint8),
                                                    np.array(pixel_changes_all_npz['moderatorEvent'], dtype=np.bool_)],
                                    dtype=np.dtype([('seconds', np.float64), 
                                                    ('xcoor', np.uint16), 
                                                    ('ycoor', np.uint16), 
                                                    ('user', np.uint32), 
                                                    ('color', np.uint8), 
                                                    ('moderator', np.bool_)])
                                                )

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