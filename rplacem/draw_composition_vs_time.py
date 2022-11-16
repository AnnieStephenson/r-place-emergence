import numpy as np
import os
import canvas_part as cp
import cProfile, pstats, io

x1 = 0
x2 = 2000
y1 = 0
y2 = 2000
part_name = 'rectangle_'+str(x1)+'.'+str(y1)+'-'+str(x2)+'.'+str(y2)  # 'txd9gh'
#canvas_comp = cp.CanvasComposition(part_name, None,data_path=os.path.join(os.getcwd(),'data'),data_file='PixelChangesCondensedData_sorted.npz')

border_path = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
canvas_area = cp.CanvasArea(border_path, None,
                            data_path=os.path.join(os.getcwd(), 'data'), data_file='PixelChangesCondensedData_sorted.npz')

time_interval = 300  # seconds
total_time = 301000  # seconds

print('save_part_over_time')
from pstats import SortKey

#pr = cProfile.Profile()
#pr.enable()
file_size_bmp, file_size_png, t_inds_list = cp.save_part_over_time(canvas_area, time_interval, total_time=total_time, part_name=part_name, delete_bmp=True, delete_png=False, show_plot=False)

cp.save_movie(os.path.join(os.getcwd(),'figs','history_'+part_name,'VsTime'), 10)
#pr.disable()
#s = io.StringIO()
#ortby = SortKey.CUMULATIVE
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
#print(s.getvalue())

cp.plot_compression(file_size_bmp, file_size_png, time_interval, total_time, part_name = part_name)
