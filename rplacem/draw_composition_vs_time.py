import numpy as np
import os
import canvas_part as cp

atlas_id_index = 'txd9gh'

pixel_changes_all = cp.get_all_pixel_changes()
print(pixel_changes_all.columns)

canvas_comp = cp.CanvasComposition(atlas_id_index, pixel_changes_all,data_path=os.path.join(os.getcwd(),'data'))
print('Canvas composition pixel changes: \n' + str(canvas_comp.pixel_changes) + '\n \n')
print('Canvas composition border path: \n' + str(canvas_comp.border_path) + '\n \n')
print('Canvas composition x coordinates: \n' + str(canvas_comp.x_coords) + '\n \n')
print('Canvas composition y coordinates: \n' + str(canvas_comp.y_coords) + '\n \n')

time_interval= 10000 #seconds
total_time = 301000 #seconds

time_inds_list_comp = cp.show_part_over_time(canvas_comp,
                                             time_interval,
                                             total_time = total_time)

print('save and compress')
file_size_bmp, file_size_png = cp.save_and_compress(canvas_comp, time_inds_list_comp)

cp.plot_compression(file_size_bmp, file_size_png, time_interval, total_time)


