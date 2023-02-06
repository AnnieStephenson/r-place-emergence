import json, os

TIME_TOTAL = 300590.208
TIME_ENLARGE1 = 99646.238
TIME_ENLARGE2 = 195583.355
TIME_WHITEONLY = 295410.186

DATA_PATH = os.path.join(os.getcwd(), 'data')
FULL_DATA_FILE = 'PixelChangesCondensedData_sorted.npz'
FIGS_PATH = os.path.join(os.getcwd(),'figs')

COLOR_DICT = json.load(open(os.path.join(DATA_PATH, 'ColorDict.json')))
WHITE = COLOR_DICT['#FFFFFF']
NUM_COLORS = len(COLOR_DICT)