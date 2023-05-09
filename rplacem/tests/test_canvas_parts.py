import sys
import numpy as np
import rplacem.canvas_part as cp
import rplacem.canvas_part_statistics as cpst
import rplacem.utilities as ut
import pytest
import pickle
import rplacem.variables_rplace2022 as var
import os

atlas_id_index = '000297'
pixel_changes_all = ut.get_all_pixel_changes()
atlas, atlas_size = ut.load_atlas()


def test_canvas_comp_instantiation():
    canvas_comp = cp.CanvasPart(id=atlas_id_index, atlas=atlas)
    np.testing.assert_equal(canvas_comp.id, atlas_id_index)


def test_canvas_comp_stat():
    canvas_comp = cp.CanvasPart(id=atlas_id_index, atlas=atlas)
    canvas_part_stat = cpst.CanvasPartStatistics(canvas_comp,
                                                    n_tbins=40,
                                                    tmax=var.TIME_TOTAL,
                                                    compute_vars={'stability': 2, 'entropy': 2,
                                                                'transitions': 2, 'attackdefense': 2},
                                                    sliding_window=14400,
                                                    returnt_binwidth=100,
                                                    trans_param=[0.2, 0.05, 7200, 10800],
                                                    timeunit=300,
                                                    dont_keep_dir=True)

    # open the previously saved canvas_part_stat
    filepath = os.path.join(var.FILE_DIR, 'tests', 'canvas_comp_stat.pickle')
    with open(filepath, 'rb') as f:
        canvas_part_stat_prev = pickle.load(f)

    np.testing.assert_equal(canvas_part_stat.stability, canvas_part_stat_prev.stability)

