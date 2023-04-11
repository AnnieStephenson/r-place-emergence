import sys
import numpy as np
import rplacem.canvas_part as cp
import rplacem.utilities as ut
import pytest

atlas_id_index = '000297'
pixel_changes_all = ut.get_all_pixel_changes()
atlas, atlas_size = ut.load_atlas()


def test_canvas_comp_instantiation():
    canvas_comp = cp.CanvasPart(id=atlas_id_index, atlas=atlas)
    np.testing.assert_equal(canvas_comp.id, atlas_id_index)
