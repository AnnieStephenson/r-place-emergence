import numpy as np
import os
import canvas_part as cp
import thermo as th
import cProfile, pstats, io
import Variables.Variables as var
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab

'''
x1 = 0
x2 = 2000
y1 = 0
y2 = 2000
part_name = 'rectangle_'+str(x1)+'.'+str(y1)+'-'+str(x2)+'.'+str(y2)  # 'txd9gh'

border_path = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
canvas_area = cp.CanvasArea(border_path, None,
                            data_path=os.path.join(os.getcwd(), 'data'), data_file='PixelChangesCondensedData_sorted.npz')
'''

file_path = os.path.join(os.getcwd(),'data', 'canvas_compositions_all.pickle') 

with open(file_path, 'rb') as f:
    canvas_parts = pickle.load(f)

'''
canvas_comp = canvas_parts[8400] #cp.CanvasComposition('000006', None,data_path=os.path.join(os.getcwd(),'data'),data_file='PixelChangesCondensedData_sorted.npz')
part_name = canvas_comp.id

'''
#print('save_part_over_time')
#from pstats import SortKey

#pr = cProfile.Profile()
#pr.enable()

'''
try:
    os.makedirs(os.path.join(os.getcwd(), 'figs', 'history_' + part_name))
except OSError: 
    print('')

try:
    os.makedirs(os.path.join(os.getcwd(), 'figs', 'history_' + part_name, 'VsTime'))
except OSError: 
    print('')
'''

#file_size_bmp, file_size_png, t_inds_list = cp.save_part_over_time(canvas_comp, 300, total_time=var.TIME_TOTAL, part_name=part_name, delete_bmp=True, delete_png=False, show_plot=False)

time_bins = 80
time_interval = var.TIME_WHITEONLY / time_bins  # seconds
time_ranges = np.arange(0, var.TIME_WHITEONLY+time_interval-1e-4, time_interval)

mean_stability = []
stability_vs_time = []
for i in range(0,len(canvas_parts)):
    if i%100 == 0:
        print(i)
    mean_stability.append( th.stability(canvas_parts[i],np.asarray([0, var.TIME_WHITEONLY]),False,False)[0] )

    stability_vs_time.append( th.stability(canvas_parts[i], time_ranges, False,False)[0] )
    #print(mean_stability[i])


data_path = os.path.join(os.getcwd(),'data')
file_path = os.path.join(data_path, 'stability_all_canvas_compositions.pickle')

with open(file_path, 'wb') as handle:
    pickle.dump([mean_stability,stability_vs_time],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)

'''
n, bins, patches = plt.hist(mean_stability, 75, range=[0,1], facecolor='blue', alpha=0.5, align='mid')

sns.despine()
plt.xlabel('mean stability')
plt.ylabel('number of compositions')
plt.xlim([0, 1])

plt.savefig(os.path.join(os.getcwd(), 'figs','mean_stability_of_all_compositions.png'))
'''

'''
#time_ranges = np.append( np.arange(0, var.TIME_WHITEONLY, time_interval) , var.TIME_WHITEONLY)
time_ranges = np.arange(0, var.TIME_WHITEONLY+time_interval-1e-4, time_interval)
stab_vs_time = th.stability(canvas_comp, time_ranges, False, False)
time_ranges2 = np.arange(0, var.TIME_WHITEONLY+time_interval*4-1e-4, time_interval*4)
stab_vs_time2 = th.stability(canvas_comp, time_ranges2, False, False)
time_ranges3 = np.arange(0, var.TIME_WHITEONLY+time_interval*10-1e-4, time_interval*10)
stab_vs_time3 = th.stability(canvas_comp, time_ranges3, False, False)
time_ranges4 = np.arange(0, var.TIME_WHITEONLY+time_interval/3-1e-4, time_interval/3)
stab_vs_time4 = th.stability(canvas_comp, time_ranges4, False, False)

plt.figure()
plt.plot(time_ranges[:-1]+time_interval/2, stab_vs_time)# 1-(1-stab_vs_time)*100)
plt.plot(time_ranges2[:-1]+time_interval*4/2, stab_vs_time2)#1-(1-stab_vs_time2)*25)
plt.plot(time_ranges3[:-1]+time_interval*10/2, stab_vs_time3)#1-(1-stab_vs_time3)*10)
plt.plot(time_ranges4[:-1]+time_interval/3/2, stab_vs_time4)#1-(1-stab_vs_time4)*300)
sns.despine()
plt.ylabel('stability')
plt.xlabel('Time [s]')

plt.savefig(os.path.join(os.getcwd(), 'figs', 'history_' + part_name, '_stability_vs_time_'+str(time_bins)+'timeRanges.png'))
'''


#cp.save_movie(os.path.join(os.getcwd(),'figs','history_'+part_name,'VsTime'), 10)

#pr.disable()
#s = io.StringIO()
#sortby = SortKey.CUMULATIVE
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
#print(s.getvalue())

#cp.plot_compression(file_size_bmp, file_size_png, time_interval, total_time, part_name = part_name)
