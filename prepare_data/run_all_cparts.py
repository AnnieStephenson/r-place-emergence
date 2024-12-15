#year = 2022
#rplacem.set_year(year)
from rplacem import var as var
import rplacem.canvas_part as cp
import rplacem.utilities as util
import pickle
import os

atlas, atlas_size = util.load_atlas()
print(atlas_size)
pixel_changes_all = util.get_all_pixel_changes()
canvas_comps = []
print(var.year, flush=True)
for i in range(0, atlas_size):
    print(i, flush=True)
    atlas_info  = cp.get_atlas_border(id=atlas[i]['id'], 
                                      atlas=atlas,
                                      addtime_before=3600*12, 
                                      addtime_after=3600*4)
    for j in range(0, len(atlas_info)):
        print(j)
        canvas_comp = cp.CanvasPart(atlas_info=atlas_info[j],
                                    pixel_changes_all=pixel_changes_all)
        canvas_comps.append(canvas_comp)
    
with open(os.path.join(var.DATA_PATH, 'canvas_comps_'+ str(var.year) + '.pkl'), 'wb') as file:
    pickle.dump(canvas_comps, file)

