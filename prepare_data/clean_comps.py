import os
from rplacem import var as var
import rplacem.canvas_part as cpart

def clean_comps(year):
    cpart_filename = 'canvas_comps_' + str(year) + '.pkl'
    cpart_filename_clean = 'canvas_comps_' + str(year) + '_clean.pkl'
    cpart_file_path = os.path.join(var.DATA_PATH, cpart_filename)
    files = sorted(os.listdir(var.DATA_PATH))
    cpart_file_path_clean = os.path.join(var.DATA_PATH, 'canvas_comps_' + str(year) + '_clean.pkl')

    if cpart_filename_clean not in files:
        if cpart_filename in files:
            print('cleaning ' + str(year), flush=True)
            cpart.clean_all_compositions(cpart_file_path, cpart_file_path_clean)
    else:
        print('Canvas parts for year ' + str(year) + ' already cleaned.')

clean_comps(2022)

clean_comps(2023)
