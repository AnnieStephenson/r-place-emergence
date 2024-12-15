import pickle
import os
import fnmatch

def combine_comps(year, pattern_var=''):
    directory_path = os.path.join(os.getcwd())
    save_path = os.path.join(os.getcwd(), 'combined_cleaned')
    filename_combined  = 'cpart_stats_' + pattern_var  + '_' + str(year) + '_all.pkl'
    pattern = 'cpart_stats_' + pattern_var + '*' + str(year) + '.pkl'

    combined_file_path = os.path.join(save_path, filename_combined)
    files = sorted(os.listdir(directory_path))

    if filename_combined not in files:
        pickle_files = [file for file in files if fnmatch.fnmatch(file, pattern)]
        if len(pickle_files)==0:
            print('No files for ' + pattern + ' to combine', flush=True)
        else:
            print('Combining files for ' + str(year) + '...', flush=True)
            cpart_stats_all=[]
            for pickle_file in pickle_files:
                print('Adding file ' + str(pickle_file) + '...', flush=True)
                file_path = os.path.join(directory_path, pickle_file)
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    cpart_stats_all = cpart_stats_all + data

                with open(combined_file_path, 'wb') as file:
                    pickle.dump(cpart_stats_all, file)
    else:
        print('Combined file for already exists for: ' + pattern, flush=True)

pattern_vars = ['sw3_ta0.35_tr1.01', 'sw3_ta0.35_tr2', 'sw3_ta0.35_tr3', 'sw3_ta0.35_tr4', 'sw3_ta0.35_tr5', 'sw3_ta0.35_tr6']
years = [2022, 2023]

for i in range(len(years)):
    for j in range(len(pattern_vars)):
        combine_comps(years[i], pattern_var=pattern_vars[j])
