from operator import mod
from re import T
import cv2
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
import PIL as pil
import seaborn as sns
import numpy as np
import os
import json


class CanvasPart(object):
    '''
    superclass containing CanvasComposition and CanvasArea. A CanvasPart object is defined by a spatial border, 
    which can be any shape or size (up to the size of the full canvas)

    attributes
    ----------
    border_path: 
    x_coords:
    y_coords:
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
    def __init__(self,
                 border_path,
                 pixel_changes_all,
                 data_path = os.path.join(os.getcwd(),'data'),
                 show_coords = False):

        self.border_path = border_path
        self.get_bounded_coords(show_coords = show_coords)
        self.find_pixel_changes_boundary(pixel_changes_all)

    def get_bounded_coords(self, show_coords =False):
        '''
        plot paths
        
        parameters
        ----------
        show_coords: tells you whether to plot
        
        returns
        --------
        
        '''
        img = np.ones((2000,2000))
        mask = np.zeros((2000,2000))
        cv2.fillPoly(mask, pts=[self.border_path], color = [1,1,1])
        masked_img = cv2.bitwise_and(img, mask)

        if show_coords:
            plt.figure()
            plt.imshow(masked_img)

        y_coords_boundary, x_coords_boundary = np.where(masked_img==1)

        self.y_coords = y_coords_boundary
        self.x_coords = x_coords_boundary
    
    def find_pixel_changes_boundary(self, pixel_changes_all, data_path=os.path.join(os.getcwd(),'data')):     
            '''
            
            parameters
            ----------
            x_coords_boundary: all the x_coords within the CanvasPart boundary
            y_coords_boundary: all the y_coords within the CanvasPart boundary
            pixel_changes: pandas dataframe of all the data
            
            returns
            -------
            pixel_changes_boundary: 
            
            '''

            x_coords_change = np.array(pixel_changes_all['x_coord'])
            y_coords_change = np.array(pixel_changes_all['y_coord'])
            color_index_changes = np.array(pixel_changes_all['color_index'])
            seconds = np.array(pixel_changes_all['seconds'])
            user_index = np.array(pixel_changes_all['user_index'])

            pixel_change_index = np.where(np.isin(x_coords_change, self.x_coords) & np.isin(y_coords_change, self.y_coords))[0]
            
            color_index_changes_boundary = color_index_changes[pixel_change_index]
            y_coord_change_boundary = y_coords_change[pixel_change_index].astype(int) 
            x_coord_change_boundary = x_coords_change[pixel_change_index].astype(int)
            time_change_boundary = seconds[pixel_change_index]
            user_id_change_boundary = user_index[pixel_change_index]

            # first look up the actual color in the dictionary to get hex
            # then need to convert from hex to RGB
            # need to make the conversion vectorized
            color_dict_path = os.path.join(data_path,'ColorsFromIdx.json')
            color_dict_file = open(color_dict_path)
            color_dict = json.load(color_dict_file)
            color_hex = np.array(list(color_dict.values()))
            color_change_boundary_hex = color_hex[color_index_changes_boundary]

            (colors_composition_change_r,
            colors_composition_change_g,
            colors_composition_change_b) = hex_to_rgb(color_change_boundary_hex)

            # now make a new dataframe with the pixel change data
            pixel_changes_boundary = pd.DataFrame(data={'seconds': time_change_boundary, 
                                                    'x_coord': x_coord_change_boundary, 
                                                    'y_coord': y_coord_change_boundary,
                                                    'user_id': user_id_change_boundary, 
                                                    'color_R': colors_composition_change_r,
                                                    'color_G': colors_composition_change_g,
                                                    'color_B': colors_composition_change_b})


            self.pixel_changes = pixel_changes_boundary 

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
    def __init__(self, 
                 id, 
                 pixel_changes_all, 
                 data_path = os.path.join(os.getcwd(),'data'), 
                 show_coords = False):

        paths, path0 = self.get_atlas_border(id, data_path=data_path)

        super().__init__(path0, pixel_changes_all)
        

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
    # TODO: vectorize this by splitting into chars and indexing for 2d array

    Turns hex color string to rgb 
    assumes there is no # in front of the hex code
    
    parameters
    ----------
    hex_str: 1d array of strings 
        vector of hex strings indicating color
    
    returns
    -------
    rgb: tuple
        tuple with the 3 coordinates corresponding to R, G, B
    '''
    hex_str = np.char.lstrip(hex_str,'#')
    rgb = np.zeros((3,len(hex_str)))
    for j in range(0,len(hex_str)):
        len_hex = len(hex_str[j])
        rgb[:,j]= [int(hex_str[j][i:i + len_hex // 3], 16) for i in range(0, len_hex, len_hex // 3)]
    return rgb[0,:], rgb[1,:], rgb[2,:]

def show_canvas_part(pixel_changes, ax=None):  
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
            img_r[pixel_changes['x_coord'].astype(int), pixel_changes['y_coord'].astype(int)] = pixel_changes['color_R']/255
            img_g[pixel_changes['x_coord'].astype(int), pixel_changes['y_coord'].astype(int)] = pixel_changes['color_G']/255
            img_b[pixel_changes['x_coord'].astype(int), pixel_changes['y_coord'].astype(int)] = pixel_changes['color_B']/255
            img = np.swapaxes(np.array([img_r, img_g, img_b]), 0, 2)

            if ax==None:
                plt.figure(origin='upper')
                plt.imshow(img,origin='upper')
            else:
                ax.imshow(img, origin='upper')

def show_part_over_time(canvas_part, 
                        time_interval, # in seconds
                        total_time= 297000 # in seconds
                        ):
        '''
        
        parameters
        ----------
        time_interval: in seconds
        
        returns
        -------
        '''
        timestamp = canvas_part.pixel_changes['seconds']
    
        num_time_steps = int(np.ceil(total_time/time_interval))

        ncols = np.min([num_time_steps, 10])
        nrows = np.max([1, int(math.ceil(num_time_steps/10))])

        #fig = plt.figure(figsize=(10,nrows)) # height corresponds in inches to number of rows. auto dpi is 100
        #gs = fig.add_gridspec(nrows, ncols, hspace=0.05, wspace=0.05)
        fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        
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
                # find the indices up to where time is at the current step
                time_inds = np.where(timestamp<=i*time_interval)[0]
                time_inds_list.append(time_inds)
                pixel_changes_time_integrated = canvas_part.pixel_changes.iloc[time_inds,:]

                show_canvas_part(pixel_changes_time_integrated, ax = ax_single)
                
            if colcount<9:
                colcount+=1
            else:
                colcount=0
                rowcount+=1
    
            x_min = np.min(canvas_part.border_path[:,0])
            x_max = np.max(canvas_part.border_path[:,0])
            y_min = np.min(canvas_part.border_path[:,1])
            y_max = np.max(canvas_part.border_path[:,1])
            plt.xlim([x_min, x_max])
            plt.ylim([y_max, y_min])

        return time_inds_list

def save_and_compress(canvas_part, time_inds_list, bmp=True, png=True):

    path_coords = canvas_part.border_path
    x_min = np.min(path_coords[:,0])
    x_max = np.max(path_coords[:,0])
    y_min = np.min(path_coords[:,1])
    y_max = np.max(path_coords[:,1])
    
    file_size_bmp = np.zeros(len(time_inds_list))
    file_size_png = np.zeros(len(time_inds_list))
    for i in range(0,len(time_inds_list)):
        im = pil.Image.new("RGB",(x_max-x_min + 1, y_max-y_min + 1),"white")
        pixel_changes_time_integrated = canvas_part.pixel_changes.iloc[time_inds_list[i],:]
        x_change_coords_integrated =  np.array(pixel_changes_time_integrated['x_coord'])
        y_change_coords_integrated =  np.array(pixel_changes_time_integrated['y_coord'])
        color_changes_integrated_r =  np.array(pixel_changes_time_integrated['color_R'])
        color_changes_integrated_g =  np.array(pixel_changes_time_integrated['color_G'])
        color_changes_integrated_b =  np.array(pixel_changes_time_integrated['color_B'])

        pixels = im.load()
        colors = np.vstack((color_changes_integrated_r, 
                            color_changes_integrated_g, 
                            color_changes_integrated_b))
        for j in range(0, len(x_change_coords_integrated)):
            pixels[int(x_change_coords_integrated[j]-x_min), 
                int(y_change_coords_integrated[j]-y_min)] = tuple(colors.transpose()[j].astype(int))
        
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

def plot_compression(file_size_bmp, file_size_png, time_interval, total_time):
    time = np.arange(time_interval, total_time + time_interval, time_interval)

    plt.figure()
    plt.plot(time, file_size_bmp, label='bmp')
    plt.plot(time, file_size_png, '--', label='png')
    sns.despine()
    plt.legend(frameon=False)
    plt.ylabel('file size (bytes)')
    plt.xlabel('time (s)')
    
    plt.figure()
    plt.plot(time, file_size_png/file_size_bmp)
    sns.despine()
    plt.ylabel('file size ratio')
    plt.xlabel('time (s)')

def get_all_pixel_changes(data_file= 'PixelChangesCondensedData_sorted.npz', data_path=os.path.join(os.getcwd(),'data')):
        '''
        TODO: deal with this
        need to rewrite and just have this function basically read in the data file
        and make it a pandas array and return that array? and then have get_pixel_changes_boundary call this function?
        Maybe this should actually be outside the class. Because we don't want to have to reload the data for each canvas section, right?
        parameters
        ----------
        data_file: .npz file
        data_path: location of data, assumed to be in cwd in 'data' folder
        
        returns
        -------
        pixel_changes: dataframe with 'timestamp', 'x_coord', 'y_coord', 'user_id', 'color'

        '''
        pixel_changes_all_npz = np.load(os.path.join(data_path, data_file))

        # load all the arrays
        seconds = pixel_changes_all_npz['seconds']
        x_coord = pixel_changes_all_npz['pixelXpos']
        y_coord = pixel_changes_all_npz['pixelYpos']
        user_index = pixel_changes_all_npz['userIndex']
        color_index = pixel_changes_all_npz['colorIndex']
        moderator_event = pixel_changes_all_npz['moderatorEvent']

        # return it as a pandas dataframe
        pixel_changes_all = pd.DataFrame(data={'seconds': seconds, 
                                               'x_coord': x_coord, 
                                               'y_coord': y_coord,
                                               'user_index': user_index, 
                                               'color_index': color_index,
                                               'moderator_event': moderator_event})

        return pixel_changes_all