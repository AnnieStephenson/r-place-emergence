import numpy as np
import os
import canvas_part as cp

part_name = 'rectangle0.0-100.100' #'txd9gh'
#canvas_comp = cp.CanvasComposition(part_name, None,data_path=os.path.join(os.getcwd(),'data'),data_file='PixelChangesCondensedData_sorted.npz')

border_path = np.array([[0,0],[0,100],[100,100],[100,0]])
canvas_area = cp.CanvasArea(border_path,None,data_path=os.path.join(os.getcwd(),'data'),data_file='PixelChangesCondensedData_sorted.npz')

time_interval= 600 #seconds
total_time = 301000 #seconds

print('save_part_over_time_simple')
file_size_bmp, file_size_png = cp.save_part_over_time_simple(canvas_area, time_interval,total_time = total_time, part_name = part_name)

cp.plot_compression(file_size_bmp, file_size_png, time_interval, total_time, part_name = part_name)


