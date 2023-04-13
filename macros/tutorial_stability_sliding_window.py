import numpy as np
import rplacem.canvas_part as cp
import rplacem.utilities as util
import rplacem.canvas_part_statistics as stat
import rplacem.compute_variables as compute
import rplacem.variables_rplace2022 as var
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import pickle

# Set parameters.
atlas_id_index = '000006'
n_tbins = 750
n_tbins_trans = 750
sliding_window_time = 10 * 3600  # 10 hrs
t_lims = np.linspace(0, var.TIME_WHITEONLY, n_tbins + 1)

# Instantiate a canvas composition object.
canvas_comp = cp.CanvasPart(id=atlas_id_index)

# Instantiate a canvas composition statistics object.
canvas_comp_stat = stat.CanvasPartStatistics(canvas_comp,
                                             n_tbins=n_tbins,
                                             n_tbins_trans=n_tbins_trans,
                                             compute_vars={'stability': 2,
                                                           'mean_stability': 0,
                                                           'entropy': 1,
                                                           'transitions': 0,
                                                           'attackdefense': 1})

# Use stability function to calculate the stable reference images (pixels1)
[stab_vst, _,
 pixels1,
 _, _, _] = compute.stability(canvas_comp,
                              t_lims=t_lims,
                              compute_average=True,
                              create_images=True,
                              sliding_window_time=sliding_window_time,
                              save_images=True,
                              print_progress=False)

# Use stability function again to get the number of differing pixels
[stab_vst,
 _, _, _, _,
 diff_pixels_vst] = compute.stability(canvas_comp,
                                      t_lims=t_lims,
                                      compute_average=True,
                                      prev_stab_col=pixels1,
                                      save_images=True)
