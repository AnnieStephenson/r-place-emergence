import json, os
from PIL import ImageColor
import numpy as np


class GlobalVars(object):
    '''
    Object containing global-level information on the canvas for a given year

    attributes
    ----------

    methods
    -------
    '''

    def __init__(self,
                 year
                 ):

        self.year = year
        self.TIME_TOTAL = 300590.208 if year == 2022 else 463089.865

        self.TIME_WHITEOUT = 295410.186 if year == 2022 else 459028.647 # only white pixel changes allowed
        self.TIME_GREYOUT = self.TIME_WHITEOUT if year == 2022 else 456227.910 # only white, black, and 3 nuances of grey allowed
        self.TIME_16COLORS = 0 if year == 2022 else 116999.080
        self.TIME_24COLORS = 99634.706 if year == 2022 else 277192.251
        self.TIME_32COLORS = 195574.105 if year == 2022 else 361992.813

        self.N_ENLARGE = 3 if year == 2022 else 7
        self.TIME_ENLARGE = np.array([0, 99646.238, 195583.355]) if year == 2022 else \
                            np.array([0, 97208.816, 155753.841, 208964.082, 277200.318, 305993.977, 362003.143])

        # shape N_ENLARGE, 2 (x or y), 2 (min or max). The given min or max values are inclusive.
        self.CANVAS_MINMAX = np.array([[[0, 999], [0, 999]], # initial canvas [[xmin, xmax], [ymin, ymax]]
                                       [[0, 1999], [0, 999]], # x right
                                       [[0, 1999], [0, 1999]]] # y bottom
                                       ) if year == 2022 else \
                             np.array([[[-500, 499], [-500, 499]], # initial canvas [[xmin, xmax], [ymin, ymax]]
                                       [[-500, 999], [-500, 499]], # x right
                                       [[-1000, 999], [-500, 499]], # x left
                                       [[-1000, 999], [-1000, 499]], # y top
                                       [[-1000, 999], [-1000, 999]], # y bottom
                                       [[-1500, 999], [-1000, 999]], # x left
                                       [[-1500, 1499], [-1000, 999]]]) # x right

        self.COOLDOWN_MIN = 250

        self.FILE_DIR = os.path.dirname(os.path.realpath(__file__))

        self.DATA_PATH = os.path.join(self.FILE_DIR, '..', 'data', str(self.year))
        self.FULL_DATA_FILE = 'PixelChangesCondensedData_sorted.npz'
        self.FIGS_PATH = os.path.join(self.FILE_DIR, '..', 'figs', str(self.year))

        self.COLOR_TO_IDX = json.load(open(os.path.join(self.DATA_PATH, 'ColorDict.json')))
        self.IDX_TO_COLOR = json.load(open(os.path.join(self.DATA_PATH, 'ColorsFromIdx.json')))
        self.WHITE = self.COLOR_TO_IDX['#FFFFFF']
        self.NUM_COLORS = len(self.COLOR_TO_IDX)

        # set color dictionaries translating the color index to the corresponding rgb or hex color
        self.COLIDX_TO_HEX = np.empty([self.NUM_COLORS], dtype='object')
        self.COLIDX_TO_RGB = np.empty([self.NUM_COLORS, 3], dtype='int32')
        for (k, v) in self.IDX_TO_COLOR.items(): # make these dictionaries numpy arrays, for more efficient use
            self.COLIDX_TO_HEX[int(k)] = v
            self.COLIDX_TO_RGB[int(k)] = np.asarray(ImageColor.getrgb(v))

var = GlobalVars(year=2022)