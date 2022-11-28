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
file_path = os.path.join(data_path, 'canvas_compositions_all.pickle')
canvas_comps = []

composition_classification_file = open( os.path.join(data_path,'atlas.json') )
atlas = json.load(composition_classification_file)
atlas_num = len(atlas)

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

for i in range(0,atlas_num):
    if i%100 == 0:
        print(i)
    canvas_comps.append( cp.CanvasComposition(atlas[i]['id'], pixel_changes_all=pixel_changes_all, atlas=atlas) )

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
    