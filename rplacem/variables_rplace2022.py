import json, os
from PIL import ImageColor
import numpy as np

TIME_TOTAL = 300590.208
TIME_ENLARGE1 = 99646.238
TIME_ENLARGE2 = 195583.355
TIME_WHITEONLY = 295410.186

DATA_PATH = os.path.join(os.getcwd(), 'data')
FULL_DATA_FILE = 'PixelChangesCondensedData_sorted.npz'
FIGS_PATH = os.path.join(os.getcwd(),'figs')

COLOR_TO_IDX = json.load(open(os.path.join(DATA_PATH, 'ColorDict.json')))
IDX_TO_COLOR = json.load(open(os.path.join(DATA_PATH, 'ColorsFromIdx.json')))
WHITE = COLOR_TO_IDX['#FFFFFF']
NUM_COLORS = len(COLOR_TO_IDX)

# set color dictionaries translating the color index to the corresponding rgb or hex color
COLIDX_TO_HEX = np.empty([NUM_COLORS], dtype='object')
COLIDX_TO_RGB = np.empty([NUM_COLORS, 3], dtype='int32')
for (k, v) in IDX_TO_COLOR.items(): # make these dictionaries numpy arrays, for more efficient use
    COLIDX_TO_HEX[int(k)] = v
    COLIDX_TO_RGB[int(k)] = np.asarray(ImageColor.getrgb(v))
