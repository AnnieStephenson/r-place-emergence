from operator import mod
from re import T
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import json
import pandas as pd
import math
import PIL as pil
import seaborn as sns

class CanvasPart(object):
    '''
    superclass containing CanvasComposition and CanvasArea. A CanvasPart object is defined by a spatial border, 
    which can be any shape or size (up to the size of the full canvas)

    attributes
    ----------
    border_path: 
    pixel_changes: 2d pandas array
        columns are timestamp, user_id, x_coord, y_coord, color
    
    methods
    -------
    draw_border(): 
        plots the empty border of the CanvasPart.
    draw_over_time():
        plots the CanvasPart pixel changes over a given time interval
    calc_file_size_raw_compressed():
        calculates the file size of the raw bmp image and the losslessly compressed png image
        at given time intervals
    '''
    def __init__(self, border_path, pixel_changes):
        self.border_path = border_path
        self.pixel_changes = pixel_changes

    def draw_path(self, path0, show_path =False):
        '''
        plot paths
        
        parameters
        ----------
        path0: list of points describing contour of selected composition
        
        returns
        --------
        
        '''
        img = np.ones((1000,1000))
        mask = np.zeros((1000,1000))
        cv2.fillPoly(mask, pts=[path0], color = [1,1,1])
        masked_img = cv2.bitwise_and(img, mask)

        if show_path==True:
            plt.figure()
            plt.imshow(masked_img)

        y_composition_coords, x_composition_coords = np.where(masked_img==1)
        return x_composition_coords, y_composition_coords


    def __get_all_pixel_change_coords(self, data_file, data_path=os.path.join(os.getcwd(),'data')):
        '''
        
        parameters
        ----------
        
        returns
        -------
        pixel_image: image of the pixel values
        '''
        print('data file: ' + data_file + '\n')
        data_file_path = os.path.join(data_path, data_file)
        tiles = pd.read_csv(data_file_path)
        
        # get coordinate array that includes all the coordinates of changed pixels from the .csv file
        # colors array includes all the hex color values of changed pixels from the .csv file
        color_changes_hex = np.array(tiles['pixel_color']) 
        coords_change_string_array = np.array(tiles['coordinate'])
        timestamp = np.array(tiles['timestamp'])
        user_id = np.array(tiles['user_id']) 
        coords_change_string_array=coords_change_string_array.astype(str)

        # deal with the admin-created rectangle coordinates    
        coords_change_split_xy = np.chararray.split(coords_change_string_array,sep=',')
        coords_change_split_xy_array = np.array([np.array(i) for i in coords_change_split_xy], dtype=object)
        coords_change_2d_array = np.zeros((coords_change_split_xy_array.shape[0],2))
        
        for i in range(0,  len(coords_change_split_xy_array)):
            if coords_change_split_xy_array[i].size>2:
                rect_corner_coords = coords_change_split_xy_array[i].astype(int)
                rect_range_x = np.arange(rect_corner_coords[0], rect_corner_coords[2]+1)
                rect_range_y = np.arange(rect_corner_coords[1], rect_corner_coords[3]+1)
                rect_coords_x, rect_coords_y = np.meshgrid(rect_range_x, rect_range_y)
                rect_coords_x = rect_coords_x.flatten()
                rect_coords_y = rect_coords_y.flatten()
                rect_coords = np.transpose(np.array([rect_coords_x, rect_coords_y]))
                coords_change_2d_array = np.insert(coords_change_2d_array, i, rect_coords, axis=0)
                
                mod_rect_color_hex = np.repeat('#FFFFFF', rect_coords.shape[0])
                color_changes_hex = np.insert(color_changes_hex, i, mod_rect_color_hex)

                mod_user_id = np.repeat(user_id[i], rect_coords.shape[0])
                user_id = np.insert(user_id, i, mod_user_id)

                mod_timestamp = np.repeat(timestamp[i], rect_coords.shape[0])
                timestamp = np.insert(timestamp, i, mod_timestamp)
            else:
                coords_change_2d_array[i] = coords_change_split_xy_array[i].astype(int)
                
        # split up into x and y for user 
        x_change_coords = coords_change_2d_array[:,0]
        y_change_coords = coords_change_2d_array[:,1]

        # put these into a pandas df: sorted timestamp, x_coord, y_coord, user_id, color_hex
        # sort the df by datetime
        # return the df
        timestamp = pd.to_datetime(list(timestamp), infer_datetime_format=True)
        file_pixel_changes = pd.DataFrame(data={'timestamp': timestamp, 
                                                'x_coord': x_change_coords, 
                                                'y_coord': y_change_coords,
                                                'user_id': user_id, 
                                                'color_hex': color_changes_hex})
        
        
        return file_pixel_changes 
    
    def get_composition_colors(self, x_composition_coords, y_composition_coords, file_pixel_changes):     
            '''
            
            parameters
            ----------
            
            returns
            --------
            
            '''
            x_change_coords = np.array(file_pixel_changes['x_coord'])
            y_change_coords = np.array(file_pixel_changes['y_coord'])
            color_changes_hex = np.array(file_pixel_changes['color_hex'])
            timestamp = np.array(file_pixel_changes['timestamp'])
            user_id = np.array(file_pixel_changes['timestamp'])

            composition_change_index = np.where(np.isin(x_change_coords, x_composition_coords) & np.isin(y_change_coords, y_composition_coords))[0]
            
            colors_composition_change_hex = color_changes_hex[composition_change_index]
            y_composition_change_coords = y_change_coords[composition_change_index].astype(int) 
            x_composition_change_coords = x_change_coords[composition_change_index].astype(int)
            timestamp_composition_change = timestamp[composition_change_index]
            timestamp_composition_change = pd.to_datetime(list(timestamp_composition_change), infer_datetime_format=True)
            user_id_composition_change = user_id[composition_change_index]


            colors_composition_change_r = np.zeros(len(colors_composition_change_hex))
            colors_composition_change_g = np.zeros(len(colors_composition_change_hex))
            colors_composition_change_b = np.zeros(len(colors_composition_change_hex))

            for i in range(len(colors_composition_change_hex)):
                (colors_composition_change_r[i],
                colors_composition_change_g[i],
                colors_composition_change_b[i]) = hex_to_rgb(colors_composition_change_hex[i])
                
            colors_composition_change = np.array([colors_composition_change_r, colors_composition_change_g, colors_composition_change_b])

            # now make a new dataframe with the 
            composition_pixel_changes = pd.DataFrame(data={'timestamp': timestamp_composition_change, 
                                                    'x_coord': x_composition_change_coords, 
                                                    'y_coord': y_composition_change_coords,
                                                    'user_id': user_id_composition_change, 
                                                    'color_R': colors_composition_change_r,
                                                    'color_G': colors_composition_change_g,
                                                    'color_B': colors_composition_change_b})

            return composition_pixel_changes #colors_composition_change, x_composition_change_coords, y_composition_change_coords
        
        
    def show_composition(self, x_composition_change_coords, y_composition_change_coords, 
                    colors_composition_change_r,
                    colors_composition_change_g,
                    colors_composition_change_b,
                    ax=None):  
            '''
            parameters
            ----------
            
            returns
            -------
            '''
            #colors_composition_change_r, colors_composition_change_g, colors_composition_change_b = colors_composition_change_list
            img = np.zeros((1000,1000))
            img_r = np.ones((1000,1000))
            img_g = np.ones((1000,1000))
            img_b = np.ones((1000,1000))
            img_r[x_composition_change_coords.astype(int), y_composition_change_coords.astype(int)] = colors_composition_change_r/255
            img_g[x_composition_change_coords.astype(int), y_composition_change_coords.astype(int)] = colors_composition_change_g/255
            img_b[x_composition_change_coords.astype(int), y_composition_change_coords.astype(int)] = colors_composition_change_b/255
            img = np.swapaxes(np.array([img_r, img_g, img_b]), 0, 2)


            if ax==None:
                plt.figure(origin='upper')
                plt.imshow(img,origin='upper')
            else:
                ax.imshow(img, origin='upper')

    def show_composition_over_time(self, id_name, file_numbers, time_interval, 
                            total_time=82.5 # total time of rplace ~82.5 hrs need to check
                            ):
            '''
            
            parameters
            ----------
            time_interval: in hours
            
            returns
            -------
            '''
            composition_pixel_changes_combined  = get_composition_pixel_changes_over_time(id_name, file_numbers)
            timestamp = composition_pixel_changes_combined['timestamp']
            timestamp = pd.to_datetime(list(timestamp), infer_datetime_format=True)
            time_delta = timestamp-timestamp[0]
            time_delta_hrs = time_delta.seconds/3600
        
            num_time_steps = int(np.ceil(total_time/time_interval))

            ncols = np.min([num_time_steps, 10])
            nrows = np.max([1, int(math.ceil(num_time_steps/10))])
            #fig = plt.figure(figsize=(10,nrows)) # height corresponds in inches to number of rows. auto dpi is 100
            #gs = fig.add_gridspec(nrows, ncols, hspace=0.05, wspace=0.05)
            fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
            print(ax.shape)
            #ax = gs.subplots(sharex=True, sharey=True)
            rowcount = 0
            colcount = 0
            time_inds_list = []
            for i in range(1, nrows*ncols + 1):
                if len(ax.shape)==2:
                    ax_single=ax[rowcount,colcount]
                else:
                    ax_single = ax[i-1]
                ax_single.axis('off')
                if i < (num_time_steps + 1):
                    time_inds = np.where(time_delta_hrs<=i*time_interval)[0]
                    time_inds_list.append(time_inds)
                    composition_pixel_changes_time_integrated = composition_pixel_changes_combined.iloc[time_inds,:]
                    x_composition_change_coords_integrated =  np.array(composition_pixel_changes_time_integrated['x_coord'])
                    y_composition_change_coords_integrated =  np.array(composition_pixel_changes_time_integrated['y_coord'])
                    color_composition_changes_integrated_r =  np.array(composition_pixel_changes_time_integrated['color_R'])
                    color_composition_changes_integrated_g =  np.array(composition_pixel_changes_time_integrated['color_G'])
                    color_composition_changes_integrated_b =  np.array(composition_pixel_changes_time_integrated['color_B'])

                    show_composition(x_composition_change_coords_integrated,
                                y_composition_change_coords_integrated, 
                                color_composition_changes_integrated_r,
                                color_composition_changes_integrated_g,
                                color_composition_changes_integrated_b, ax = ax_single)
                    
                if colcount<9:
                    colcount+=1
                else:
                    colcount=0
                    rowcount+=1
                path0, path = get_path(id_name)
                path_coords = np.array(max(path0.values()))
                x_min = np.min(path_coords[:,0])
                x_max = np.max(path_coords[:,0])
                y_min = np.min(path_coords[:,1])
                y_max = np.max(path_coords[:,1])
                plt.xlim([x_min, x_max])
                plt.ylim([y_max, y_min])

            return (composition_pixel_changes_combined,
                    time_inds_list)
    
    def get_composition_change_coords_colors(self, id_name, data_file):
            '''
            parameters
            ----------
            
            returns
            -------
            
            '''
            
            path0 = self.border_path

            # get the coordinates of all tiles within the composition
            x_composition_coords, y_composition_coords = draw_path(path0)
            # get the coordinates of tiles that were changed in the dataset:
            file_pixel_changes = __get_all_pixel_change_coords(data_file)
            #x_change_coords, y_change_coords, color_changes_hex = get_all_pixel_change_coords(data_file)

            # get the colors the tiles were changed to: color_changes
            # and the coordinates of the composition tiles that were changed in the dataset: x_composition_change_coords, y_composition_change_coords
            composition_pixel_changes = get_composition_colors(x_composition_coords, y_composition_coords, file_pixel_changes)
                                                                            
            return composition_pixel_changes
    
    def get_composition_pixel_changes_over_time(self, id_name, file_numbers):
        '''
        
        parameters
        ----------
        
        returns
        -------
        
        '''
        composition_pixel_changes_combined = pd.DataFrame(data={'timestamp':[], 
                                                        'x_coord':[],
                                                        'y_coord': [],
                                                        'user_id':[],
                                                        'color_R':[],
                                                        'color_G':[],
                                                        'color_B':[]})

        for i in range(0, file_numbers.size):
            if file_numbers[i]<=9:
                extra_str = '0'
            else:
                extra_str = ''
            data_file = '2022_place_canvas_history-0000000000' + extra_str + str(file_numbers[i]) + '.csv'
            composition_pixel_changes = get_composition_change_coords_colors(id_name, data_file)
            composition_pixel_changes_combined = pd.concat([composition_pixel_changes_combined, composition_pixel_changes])

        # sort by datetime 
        composition_pixel_changes_combined['timestamp']=pd.to_datetime(composition_pixel_changes_combined['timestamp'], infer_datetime_format=True)
        composition_pixel_changes_combined = composition_pixel_changes_combined.sort_values(by=['timestamp']) 
            
        return composition_pixel_changes_combined

    def save_and_compress_composition(id_name, composition_pixel_changes_combined, time_inds_list, bmp=True, png=True):
        path0, path = get_path(id_name)
        path_coords = np.array(max(path0.values()))
        x_min = np.min(path_coords[:,0])
        x_max = np.max(path_coords[:,0])
        y_min = np.min(path_coords[:,1])
        y_max = np.max(path_coords[:,1])
        
        file_size_bmp = np.zeros(len(time_inds_list))
        file_size_png = np.zeros(len(time_inds_list))
        for i in range(0,len(time_inds_list)):
            im = pil.Image.new("RGB",(x_max-x_min + 1, y_max-y_min + 1),"white")
            composition_pixel_changes_time_integrated = composition_pixel_changes_combined.iloc[time_inds_list[i],:]
            x_composition_change_coords_integrated =  np.array(composition_pixel_changes_time_integrated['x_coord'])
            y_composition_change_coords_integrated =  np.array(composition_pixel_changes_time_integrated['y_coord'])
            color_composition_changes_integrated_r =  np.array(composition_pixel_changes_time_integrated['color_R'])
            color_composition_changes_integrated_g =  np.array(composition_pixel_changes_time_integrated['color_G'])
            color_composition_changes_integrated_b =  np.array(composition_pixel_changes_time_integrated['color_B'])

            pixels = im.load()
            colors = np.vstack((color_composition_changes_integrated_r, 
                                color_composition_changes_integrated_g, 
                                color_composition_changes_integrated_b))
            for j in range(0, len(x_composition_change_coords_integrated)):
                pixels[int(x_composition_change_coords_integrated[j]-x_min), 
                    int(y_composition_change_coords_integrated[j]-y_min)] = tuple(colors.transpose()[j].astype(int))
            
            if bmp:
                im.save('frames' + str(i) + '.bmp')
                with open('frames' + str(i) + '.bmp', "rb") as image_file:
                    f = image_file.read()
                    byte_array = bytearray(f)
                file_size_bmp[i] = len(byte_array)
                
            if png:
                im.save('compressed_frames' + str(i) + '.png', optimize = True)
                with open('compressed_frames' + str(i) + '.png', "rb") as image_file:
                    f = image_file.read()
                    byte_array = bytearray(f)
                file_size_png[i] = len(byte_array)
                
        return file_size_bmp, file_size_png

    def plot_compression(self, file_size_bmp, file_size_png, time_interval, total_time):
        time = np.arange(time_interval, total_time + time_interval, time_interval)

        plt.figure()
        plt.plot(time, file_size_bmp, label='bmp')
        plt.plot(time, file_size_png, '--', label='png')
        sns.despine()
        plt.legend(frameon=False)
        plt.ylabel('file size (bytes)')
        plt.xlabel('time (hrs)')
        
        plt.figure()
        plt.plot(time, file_size_png/file_size_bmp)
        sns.despine()
        plt.ylabel('file size ratio')
        plt.xlabel('time (hrs)')

class CanvasComposition(CanvasPart):
    '''
    subclass of CanvasPart. A CanvasComposition object is a particular 'composition' created by a group of users, 
    identified in the Atlas.json file created by the r/place atlas project.  

    attributes
    ----------
    id_name: string
        the string from the atlas file that identifies a particular composition
    border_path: inherited from superclass
    pixel_changes: inherited from superclass
    
    methods
    -------
    get_atlas_border(): 
    draw_border(): inherited from superclass
    calc_file_size_raw_compressed(): inherited from superclass

    '''
    def __init__(self, border_path, pixel_changes):
        super().__init__(border_path, pixel_changes)

    def get_atlas_border(self, id, data_path=os.path.join(os.getcwd(),'data')): 
        '''
        Read info from json file
        
        parameters
        ----------
        id: float or int

        returns
        -------
        path: list of points that describes the countour of the selected composition
        '''
        atlas_path = os.path.join(data_path,'atlas.json')
        composition_classification_file = open(atlas_path)
        atlas = json.load(composition_classification_file)
        
        for i in range(len(atlas)):
            if atlas[i]['id']==id:
                id_index = i
    
        paths = atlas[id_index]['path']
        path0 = np.array(list(paths.values()))[0] # first

        self.id_name = str(atlas[id_index]['name'])
        self.border_path = path0
        return paths, path0

class CanvasArea(CanvasPart):
    '''
    subclass of CanvasPart. A CanvasArea object contains the pixels within a user-specified border, which may 
    include any area of the canvas, up to the size of the full canvas. 

    attributes
    ----------
    border_path: inherited from superclass
    pixel_changes: inherited from superclass
    
    methods
    -------
    draw_border(): inherited from superclass
    draw_over_time(): inherited from superclass
    calc_file_size_raw_compressed(): inherited from superclass

    '''
    def __init__(self, border_path, pixel_changes):
        super().__init__(border_path, pixel_changes)

class ColorMovement:
    '''
    A ColorMovement object is defined by pixels of a single color that seemingly diffuse accross the canvas. We can
    characterize how this object grows and travels over time. There is no set border. 

    attributes
    ----------
    color: array-like
        single RGB value for the color of the ColorMovement.
    seed_point: tuple
        an (x,y) point that is the starting point for the ColorMovement object. Ideally, this should be the first pixel
        that appears in the ColorMovement. The other pixels in the ColorMovement are nearest neighbors of the seed_point, 
        and then nearest neighbors of those neighbors, and so on. 
    pixel_changes: 2d pandas array
        characterizes the growth and diffusion of the pixels in the ColorMovement object rather than tracking the pixel changes within a set border.
        Each pixel_change has a start and stop time to identify when it changed to match the color of the ColorMovement and when it was changed to a new color. 
        columns: timestamp_start, timestamp_end, user_id, x_coord, y_coord      
    size: 1d numpy array
        size (in pixels) of the ColorMovement object over time
    
    methods
    -------
    '''
    

def hex_to_rgb(hex_str):
    '''
    Turns hex color string to rgb 
    
    parameters
    ----------
    hex_str: string 
        hex string indicating color
    
    returns
    -------
    rgb: tuple
        tuple with the 3 coordinates corresponding to R, G, B
    '''
    hex_str = hex_str.lstrip('#')
    len_hex = len(hex_str)
    rgb = tuple(int(hex_str[i:i + len_hex // 3], 16) for i in range(0, len_hex, len_hex // 3))
    return rgb



    
