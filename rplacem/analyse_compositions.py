import numpy as np
import os
import canvas_part as cp
import thermo as th
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab
import Variables.Variables as var

file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_all.pickle') 

with open(file_path, 'rb') as f:
    canvas_parts = pickle.load(f)

# histogram of # of pixels in each composition
size_in_pixels = [cp.coords.shape[1] for cp in canvas_parts]
xmin = 3
xmax = 1.5e5
n, bins, patches = plt.hist(size_in_pixels, bins=np.logspace(np.log10(xmin), np.log10(xmax), 100), facecolor='blue', alpha=0.5, align='mid')

sns.despine()
plt.xlabel('composition size [#pixels]')
plt.ylabel('number of compositions')
plt.xscale('log')
plt.xlim([xmin,xmax])

plt.savefig(os.path.join(var.FIGS_PATH,'size_of_all_compositions.png'))