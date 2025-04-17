
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
        self.TIME_TOTAL = {2023: 463089.865, 2022: 300590.208, 2017: 259195.61}[year] # total time of the canvas in seconds
        self.TIME_WHITEOUT = {2023: 459028.647, 2022: 295410.186, 2017: self.TIME_TOTAL}[year] # only white pixel changes allowed
        self.TIME_GREYOUT = {2023: 456227.910, 2022: self.TIME_WHITEOUT, 2017: self.TIME_TOTAL}[year] # only white, black, and 3 nuances of grey allowed
        self.TIME_16COLORS = {2023: 116999.080, 2022: 0, 2017: 0}[year] # in 2017 and 2022, started with 16 colors
        self.TIME_24COLORS = {2023: 277192.251, 2022: 99634.706, 2017: self.TIME_TOTAL}[year] # in 2017, never had 24 colors 
        self.TIME_32COLORS = {2023: 361992.813, 2022: 195574.105, 2017: self.TIME_TOTAL}[year] # in 2017, never had 32 colors
        self.TIME_PARTIAL_240S_COOLDOWN = {2023: 273600, 2022: self.TIME_TOTAL, 2017: self.TIME_TOTAL}[year]# 2023 number from the reddit mod (Mikhail)
        self.TIME_180S_AND_240S_COOLDOWN = {2023: self.TIME_GREYOUT, 2022: self.TIME_TOTAL, 2017: self.TIME_TOTAL}[year] # 2023 info from the reddit mod (Mikhail)
        self.TIME_120S_COOLDOWN = {2023: 459180, 2022: self.TIME_TOTAL, 2017: self.TIME_TOTAL}[year] # 2023 number from the reddit mod (Mikhail)
        self.TIME_60S_COOLDOWN = {2023: 459420, 2022: self.TIME_TOTAL, 2017: self.TIME_TOTAL}[year]# 2023 number from the reddit mod (Mikhail)
        self.TIME_30S_COOLDOWN = {2023: 60200, 2022: self.TIME_TOTAL, 2017: self.TIME_TOTAL}[year]# 2023 number from the reddit mod (Mikhail)

        self.N_ENLARGE = {2023: 7, 2022: 3, 2017: 1}[year] # technically this is number of canvas sizes, which would be number of enlargments + 1
        self.TIME_ENLARGE =  {2022: np.array([0, 99646.238, 195583.355]),
                              2023: np.array([0, 97208.816, 155753.841, 208964.082, 277200.318, 305993.977, 362003.143]),
                              2017: np.array([0])}[year] # first time is actually just 0 
        
        # shape N_ENLARGE, 2 (x or y), 2 (min or max). The given min or max values are inclusive.
        self.CANVAS_MINMAX = {2023: np.array([[[-500, 499], [-500, 499]], # initial canvas [[xmin, xmax], [ymin, ymax]]
                                              [[-500, 999], [-500, 499]], # x right
                                              [[-1000, 999], [-500, 499]], # x left
                                              [[-1000, 999], [-1000, 499]], # y top
                                              [[-1000, 999], [-1000, 999]], # y bottom
                                              [[-1500, 999], [-1000, 999]], # x left
                                              [[-1500, 1499], [-1000, 999]]]), # x right
                              2022: np.array([[[0, 999], [0, 999]], # initial canvas [[xmin, xmax], [ymin, ymax]]
                                             [[0, 1999], [0, 999]], # x right
                                             [[0, 1999], [0, 1999]]]), # y bottom 
                              2017: np.array([[[0, 999], [0, 999]]]), # initial canvas [[xmin, xmax], [ymin, ymax]]
                                       }[year] # initial canvas [[xmin, xmax], [ymin, ymax]]
        
        self.COOLDOWN_MIN = 250

        self.FILE_DIR = os.path.dirname(os.path.realpath(__file__))

        self.DATA_PATH = os.path.join(self.FILE_DIR, '..', 'data', str(self.year))
        self.FULL_DATA_FILE = 'PixelChangesCondensedData_sorted.npz'
        self.FIGS_PATH = os.path.join(self.FILE_DIR, '..', 'figs', str(self.year))

        self.COLOR_TO_IDX = json.load(open(os.path.join(self.DATA_PATH, 'ColorDict.json')))
        self.IDX_TO_COLOR = json.load(open(os.path.join(self.DATA_PATH, 'ColorsFromIdx.json')))
        self.WHITE = self.COLOR_TO_IDX['#FFFFFF']

        black_hex = {2023: '#000000', 2022: '#000000', 2017: '#222222'}[year] # 2017 did not have a true black
        self.BLACK = self.COLOR_TO_IDX[black_hex] # 2017 did not have a true black
        purple_hex = {2023: '#811E9F', 2022: '#811E9F', 2017: '#820080'}[year]
        self.PURPLE = self.COLOR_TO_IDX[purple_hex]
        self.NUM_COLORS = len(self.COLOR_TO_IDX)

        # set color dictionaries translating the color index to the corresponding rgb or hex color
        self.COLIDX_TO_HEX = np.empty([self.NUM_COLORS], dtype='object')
        self.COLIDX_TO_RGB = np.empty([self.NUM_COLORS, 3], dtype='int32')
        for (k, v) in self.IDX_TO_COLOR.items(): # make these dictionaries numpy arrays, for more efficient use
            self.COLIDX_TO_HEX[int(k)] = v
            self.COLIDX_TO_RGB[int(k)] = np.asarray(ImageColor.getrgb(v))

var = GlobalVars(year=2022)

def set_year(year):
    global var
    var = GlobalVars(year=year)