from operator import mod
from re import T
import cv2
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
from PIL import Image,ImageColor
import seaborn as sns
import numpy as np
import os
import json
import pickle
import sys
import glob
import cProfile

class CanvasPart(object):
    '''
    superclass with subclasses CanvasComposition and CanvasArea. A CanvasPart 
    object is a defined "part" of a canvas with a set spatial border, which 
    can be any shape or size (up to the size of the full canvas) 

    attributes
    ----------
    border_path : 2d numpy array with shape (number of points in path, 2)
        Array of x,y coordinates defining the boundary of the CanvasPart
    x_coords : 1d numpy array
        x-coordinates of all the pixels contained inside the CanvasPart
    y_coords : 1d numpy array
        y0coordinates of all the pixels contained inside the CanvasPart
    pixel_changes : 2d pandas array
        Pixel changes over time within the CanvasPart boundary. 
        Columns are seconds, x_coord, y_coord, user_id, color_R, color_G, 
        color_B
    colidx_to_hex : dictionary
        dictionary with keys of color index and values of color hex code
    colidx_to_rgb : dictionary
        dictionary with keys of color index and values of color rgb
    data_path : string
        full path where the pixel data is stored
    data_file : string
        name of the pixel data file
    
    methods
    -------
    get_bounded_coords(self, show_coords =False)
        Finds the x and y coordinates within the boundary of the 
        CanvasPart object. Sets the x_coord and y_coord attributes
        of the CanvasPart object
    find_pixel_changes_boundary(self, 
                                pixel_changes_all, 
                                data_path=os.path.join(os.getcwd(),'data'))
        Find all the pixel changes within the boundary of the CanvasPart object
        and set the pixel_changes attribute of the CanvasPart object accordingly
    set_color_dictionaries(self)
        Set up the dictionaries translating the color index to the right rgb or hex color
    get_rgb(self, col_idx)
        Get (r,g,b) triplet from the color index
    '''
    def __init__(self,
                 border_path,
                 pixel_changes_all=None,
                 data_path=os.path.join(os.getcwd(),'data'),
                 data_file='PixelChangesCondensedData_sorted.npz',
                 show_coords=False):
        '''
        Constructor for CanvasPart object

        Parameters
        ----------
        border_path : 2d numpy array with shape (number of points in path, 2)
            Array of x,y coordinates defining the boundary of the CanvasPart
        pixel_changes_all : 2d pandas dataframe or None, optional
            Contains all of the pixel change data from the entire dataset. If ==None, then the whole dataset is loaded when the class instance is initialized.
        data_path : str, optional
            Path to where the pixel data file is stored
        data_file: str, optional
             name of the pixel data file
        show_coords : bool, optional
            If True, plots the mask of the CanvasPart boundary
        '''
        self.border_path = border_path
        self.x_coords = None
        self.y_coords = None
        self.pixel_changes = None 
        self.colidx_to_hex = {}
        self.colidx_to_rgb = {}
        self.data_path = data_path
        self.data_file = data_file

        # set the color (hex and rgb) dictionaries
        self.set_color_dictionaries()
    
        # set the x_coords and y_coords within the border
        self.get_bounded_coords(show_coords=show_coords)

        # set the pixel changes within the boundary
        self.find_pixel_changes_boundary(self.pixel_changes)
        
    def set_color_dictionaries(self):
        ''' set the color dictionaries from the data file '''

        color_dict_file = open(os.path.join(self.data_path,'ColorsFromIdx.json'))
        color_dict = json.load(color_dict_file)
        self.colidx_to_hex = {int(k) : v
                              for (k,v) in color_dict.items()}
        self.colidx_to_rgb = {k : ImageColor.getrgb(v)
                              for (k,v) in self.colidx_to_hex.items()}

    def get_rgb(self, col_idx):
        ''' Get (r,g,b) triplet from the color index '''

        if hasattr(col_idx, "__len__"):
            rgb_color = np.array([self.colidx_to_rgb[c] for c in col_idx])
        else: 
            rgb_color = self.colidx_to_rgb[col_idx]
        return rgb_color     
        
    def get_bounded_coords(self, show_coords=False):
        '''
        Finds the x and y coordinates within the boundary of the 
        CanvasPart object. Sets the x_coord and y_coord attributes
        of the CanvasPart object
        
        parameters
        ----------
        show_coords : bool, optional
            If True, plots the mask of the CanvasPart boundary
        
        '''
        # total final size of the r/place canvas
        img = np.ones((2000,2000))
        mask = np.zeros((2000,2000))

        # create mask from border_path
        cv2.fillPoly(mask, pts=[self.border_path], color=[1,1,1])
        masked_img = cv2.bitwise_and(img, mask)

        if show_coords:
            plt.figure()
            plt.imshow(masked_img)

        y_coords_boundary, x_coords_boundary = np.where(masked_img == 1)

        self.y_coords = y_coords_boundary
        self.x_coords = x_coords_boundary
    
    def find_pixel_changes_boundary(self, 
                                    pixel_changes_all
                                    ):     
            '''
            Find all the pixel changes within the boundary of the CanvasPart object
            and set the pixel_changes attribute of the CanvasPart object accordingly
            
            parameters
            ----------
            pixel_changes_all : 2d pandas dataframe 
                Contains all of the pixel change data from the entire dataset
            data_path : str, optional
                Path to where the pixel data file is stored

            '''

            if pixel_changes_all is None:
                pixel_changes_all_npz = np.load(os.path.join(self.data_path, self.data_file))

                # load all the arrays
                seconds = pixel_changes_all_npz['seconds']
                x_coords_change = pixel_changes_all_npz['pixelXpos']
                y_coords_change = pixel_changes_all_npz['pixelYpos']
                user_index = pixel_changes_all_npz['userIndex']
                color_index_changes = pixel_changes_all_npz['colorIndex']

            else:
                x_coords_change = np.array(pixel_changes_all['x_coord'])
                y_coords_change = np.array(pixel_changes_all['y_coord'])
                color_index_changes = np.array(pixel_changes_all['color_index'])
                seconds = np.array(pixel_changes_all['seconds'])
                user_index = np.array(pixel_changes_all['user_index'])
            pixel_change_index = np.where((np.isin(x_coords_change, self.x_coords) 
                                           & np.isin(y_coords_change, self.y_coords)))[0]
            
            color_index_changes_boundary = color_index_changes[pixel_change_index]
            y_coord_change_boundary = y_coords_change[pixel_change_index].astype(int) 
            x_coord_change_boundary = x_coords_change[pixel_change_index].astype(int)
            time_change_boundary = seconds[pixel_change_index]
            user_id_change_boundary = user_index[pixel_change_index]

            # now make a new dataframe with the pixel change data
            self.pixel_changes = pd.DataFrame(data={'seconds': time_change_boundary, 
                                                    'x_coord': x_coord_change_boundary, 
                                                    'y_coord': y_coord_change_boundary,
                                                    'user_id': user_id_change_boundary, 
                                                    'color_id': color_index_changes_boundary,
                                                    })


class CanvasComposition(CanvasPart):
    '''
    Subclass of CanvasPart. A CanvasComposition object is a particular 
    'composition' created by a group of users, identified in the atlas.json 
    file created by the r/place atlas project.  

    attributes
    ----------
    id : string
        The string from the atlas file that identifies a particular composition
    border_path : inherited from superclass
    pixel_changes : inherited from superclass
    
    methods
    -------
    get_atlas_border() 
        Look up the border path for the id index in the atlas.json file
    get_bounded_coords(self, show_coords=False)
        Inherited from superclass
    find_pixel_changes_boundary(self, 
                                pixel_changes_all, 
                                data_path=os.path.join(os.getcwd(),'data'))
        Inherited from superclass

    '''
    def __init__(self, 
                 id, 
                 pixel_changes_all=None, 
                 data_path = os.path.join(os.getcwd(),'data'), 
                 data_file='PixelChangesCondensedData_sorted.npz',
                 show_coords = False):

        '''
        Constructor for CanvasComposition object

        Parameters
        ----------
        id: string
            The string from the atlas file that identifies a particular composition
        pixel_changes_all : 2d pandas dataframe or None, optional
            Contains all of the pixel change data from the entire dataset
        data_path : str, optional
            Path to where the pixel data file is stored
        show_coords : bool, optional
            If True, plots the mask of the CanvasPart boundary
        '''

        # get the border_path from the atlas.json file
        # path0 is the initial path. 
        # TODO: add handling for if path changes over time
        paths, path0 = self.get_atlas_border(id, data_path=data_path)

        super().__init__(path0, 
                         pixel_changes_all=pixel_changes_all, 
                         data_path=data_path, 
                         data_file=data_file, 
                         show_coords=show_coords)
        

    def get_atlas_border(self, id, data_path=os.path.join(os.getcwd(),'data')): 
        '''
        Get the border of the CanvasComposition object from the atlas.json file
        
        parameters
        ----------
        id : string
            The string from the atlas file that identifies a particular composition
        data_path : str, optional
            Path to where the pixel data file is stored

        returns
        -------
        path : array-like
            x, y coordinates describes the boundary of the selected composition over time
        path0 : 2d numpy array
            x, y coordinates describes the boundary of the selected composition at the initial time

        '''
        atlas_path = os.path.join(data_path,'atlas.json')
        composition_classification_file = open(atlas_path)
        atlas = json.load(composition_classification_file)
        
        for i in range(len(atlas)):
            if atlas[i]['id'] == id:
                id_index = i
    
        paths = atlas[id_index]['path']
        path0 = np.array(list(paths.values())[0]) # initial path

        self.id_name = str(atlas[id_index]['name'])
        composition_classification_file.close()
        return paths, path0

class CanvasArea(CanvasPart):
    '''
    subclass of CanvasPart. A CanvasArea object contains the pixels within a 
    user-specified border, which may include any area of the canvas, up to the
    size of the full canvas. 

    attributes
    ----------
    border_path : inherited from superclass
    pixel_changes : inherited from superclass
    
    methods
    -------
    get_bounded_coords(self, show_coords =False)
        Inherited from superclass
    find_pixel_changes_boundary(self, 
                                pixel_changes_all, 
                                data_path=os.path.join(os.getcwd(),'data'))
        Inherited from superclass

    '''
    def __init__(self, 
                 border_path, 
                 pixel_changes_all,
                 data_path=os.path.join(os.getcwd(),'data'),
                 data_file='PixelChangesCondensedData_sorted.npz'):
        super().__init__(border_path, 
                         pixel_changes_all=pixel_changes_all,
                         data_path=data_path,
                         data_file=data_file)

class ColorMovement:
    '''
    A ColorMovement object is defined by pixels of a single color that 
    seemingly diffuse accross the canvas. We can characterize how this object 
    grows and travels over time. There is no set border. 

    attributes
    ----------
    color : array-like
        single RGB value for the color of the ColorMovement.
    seed_point : tuple
        an (x,y) point that is the starting point for the ColorMovement 
        object. Ideally, this should be the first pixel that appears in the 
        ColorMovement. The other pixels in the ColorMovement are nearest 
        neighbors of the seed_point, and then nearest neighbors of those 
        neighbors, and so on. 
    pixel_changes : 2d pandas array
        characterizes the growth and diffusion of the pixels in the 
        ColorMovement object rather than tracking the pixel changes within a 
        set border. Each pixel_change has a start and stop time to identify 
        when it changed to match the color of the ColorMovement and when it 
        was changed to a new color. columns: timestamp_start, timestamp_end, 
        user_id, x_coord, y_coord      
    size : 1d numpy array
        size (in pixels) of the ColorMovement object over time
    
    methods
    -------
    '''

def get_file_size(path):
    ''' Gets the length of a file in bytes'''
    f = open(path, "rb").read()
    byte_array = bytearray(f)
    return len(byte_array)

def show_canvas_part(pixels, ax=None):  
    '''
    Plots the pixels of a CanvasPart at a snapshot in time

    parameters
    ----------
    cp: CanvasPart class instance
    time_inds: array of time indices of the pixel changes taken into account in the shown canvas
    
    '''
    if ax == None:
        plt.figure(origin='upper')
        plt.imshow(pixels, origin='upper')
    else:
        ax.imshow(pixels, origin='upper')
  
def save_part_over_time(canvas_part,
                        time_interval, # in seconds
                        total_time=301000, # in seconds
                        part_name = 'cp', # only for name of output
                        delete_bmp = True,
                        delete_png = False,
                        show_plot = True
                        ):
    '''
    Saves images of the canvas part for each time step

    parameters
    ----------
    canvas_part : CanvasPart object
    time_interval : float
        time interval at which to plot (in seconds)
    total_time : float, optional
        total time to plot intervals until (in seconds)
    part_name : string, optional
        id/name of part for naming output file
    delete_bmp : boolean, optional
        if True, the .bmp files are deleted after their size is determined
    show_plot : boolean, optional
        if True, plots all the frames on a grid
    
    returns
    -------
    file_size_bmp : float
        size of png image in bytes
    file_size_png : float
        size of png image in bytes
    t_inds_list : list
        list of arrays of all the time indices of pixel changes in 
        each time interval
    '''

    seconds = np.array(canvas_part.pixel_changes['seconds'])
    xcoor = np.array(canvas_part.pixel_changes['x_coord'])
    ycoor = np.array(canvas_part.pixel_changes['y_coord'])
    color = np.array(canvas_part.pixel_changes['color_id'])

    num_time_steps = int(np.ceil(total_time/time_interval))
    file_size_bmp = np.zeros(num_time_steps)
    file_size_png = np.zeros(num_time_steps)
    
    path_coords = canvas_part.border_path
    x_min = np.min(path_coords[:,0])
    x_max = np.max(path_coords[:,0])
    y_min = np.min(path_coords[:,1])
    y_max = np.max(path_coords[:,1])

    pixels = np.full((y_max - y_min + 1, x_max - x_min + 1, 3), 255, dtype=np.uint8) # consider that [r,g,b] will be readable as (r,g,b) ##### fill as white first ##### not sure why, but the pixels should be [y,x,rgb]
    out_path = os.path.join(os.getcwd(), 'figs', 'history_' + part_name)
    try:
        os.makedirs(out_path)
    except OSError as err: # empty directory if it already exists
        for f in glob.glob(out_path + '/*'):
            os.remove(f)
    
    if show_plot:
        ncols = np.min([num_time_steps, 10])
        nrows = np.max([1, int(math.ceil(num_time_steps/10))])
        #fig = plt.figure(figsize=(10,nrows)) # height corresponds in inches to number of rows. auto dpi is 100
        #gs = fig.add_gridspec(nrows, ncols, hspace=0.05, wspace=0.05)
        fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        #ax = gs.subplots(sharex=True, sharey=True)
        rowcount = 0
        colcount = 0
    t_inds_list = []
    for t_step_idx in range(0, num_time_steps): # use the fact that arrays are time-sorted  
        
        # get the indices of the times within the interval   
        t_inds = np.where((seconds>t_step_idx*time_interval) & (seconds<(t_step_idx + 1)*time_interval))[0]
        t_inds_list.append(t_inds)
        if len(t_inds)!=0:
            pixels[ycoor[t_inds]-y_min, xcoor[t_inds]-x_min, :] = canvas_part.get_rgb(color[t_inds])

        # save image
        im = Image.fromarray(pixels)
        im_path = os.path.join(out_path, 'canvaspart_time{:06d}'.format(int(t_step_idx*time_interval)) )
        im.save(im_path + '.png')
        im.save(im_path + '.bmp')
        file_size_png[t_step_idx] = get_file_size(im_path + '.png')
        file_size_bmp[t_step_idx] = get_file_size(im_path + '.bmp')
        if delete_bmp:
            os.remove(im_path + '.bmp')
        if delete_png:
            os.remove(im_path + '.png')

        if show_plot:
            if len(ax.shape)==2:
                ax_single=ax[rowcount,colcount]
            else:
                ax_single = ax[t_step_idx]
            ax_single.axis('off')
            show_canvas_part(pixels, ax = ax_single)
                
            if colcount < 9:
                colcount += 1
            else:
                colcount = 0
                rowcount += 1
        
    return file_size_bmp, file_size_png, t_inds_list

        
def plot_compression(file_size_bmp, file_size_png, time_interval, total_time, part_name = ''):
    '''
    plot the file size ratio over time

    parameters
    ----------
    file_size_bmp : float
        size of png image in bytes
    file_size_png : float
        size of png image in bytes
    time_interval : float
        time interval at which to plot (in seconds)
    total_time : float
        total time to plot intervals until (in seconds)
    part_name : string
        for the naming of the output saved plot        
    '''

    time = np.arange(time_interval, total_time + 2*time_interval, time_interval)
    
    plt.figure()
    plt.plot(time, file_size_png/file_size_bmp)
    sns.despine()
    plt.ylabel('Computable Information Density (file size ratio)')
    plt.xlabel('Time (s)')
    plt.savefig(os.path.join(os.getcwd(),'figs', 'history_' + part_name, '_file_size_compression_ratio.png'))


def get_all_pixel_changes(data_file='PixelChangesCondensedData_sorted.npz', 
                          data_path=os.path.join(os.getcwd(),'data')):
    '''
    load all the pixel change data and put it in a pandas array for easy access
    
    parameters
    ----------
    data_file : string that ends in .npz, optional
    data_path : location of data, assumed to be in cwd in 'data' folder, optional
    
    returns
    -------
    pixel_changes_all : 2d pandas dataframe 
            Contains all of the pixel change data from the entire dataset

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
                                           'moderator_event': moderator_event
                                           })

    return pixel_changes_all

def save_canvas_part_time_steps(canvas_comp,  
                                time_inds_list_comp,
                                time_interval,
                                file_size_bmp,
                                file_size_png,
                                data_path = os.path.join(os.getcwd(),'data'),
                                file_name = 'canvas_part_data'):
    '''
    save the variables associated with the CanvasPart object 

    '''
    file_path = os.path.join(data_path, file_name + '.pickle')
    with open(file_path, 'wb') as handle:
        pickle.dump([canvas_comp, 
                    time_inds_list_comp,
                    time_interval,
                    file_size_bmp,
                    file_size_png], 
                    handle, 
                    protocol=pickle.HIGHEST_PROTOCOL)

def load_canvas_part_time_steps(data_path = os.path.join(os.getcwd(),'data'),
                                file_name = 'canvas_part_data'):
    '''
    save the variables associated with the CanvasPart object 
    '''
    file_path = os.path.join(data_path, file_name + '.pickle')
    with open(file_path,'rb') as f: 
        canvas_part_parameters = pickle.load(f)

    return canvas_part_parameters

def save_movie(image_path, 
               movie_tool='moviepy', 
               fps=1,
               codec='libx264',
               video_type='mp4'):
    '''
    Save movie of .png images in the path.
    
    parameters
    ----------
    image_path: string
        Path where .png images can be found
    movie_tool: string
        Movie handling package you want to use
        values can be 'moviepy', 'ffmpeg-python', or 'ffmpeg'
        You must have the corresponding packages installed for this to work.
        'moviepy' and 'ffmpeg-python' refer to python packages. 'ffmpeg' refers 
        to software that must be installed on your system without requiring a 
        specific python package. 
    '''
    image_files = list(np.sort(glob.glob(os.path.join(image_path,'*.png'))))
    png_name0 = os.path.basename(image_files[0][0:-4])
    movie_name = png_name0 + '_fps' + str(fps)
    movie_file = os.path.join(image_path, movie_name) + '.' + video_type
    
    if movie_tool=='moviepy':
        if 'imsc' not in sys.modules:
            import moviepy.video.io.ImageSequenceClip as imsc
        clip = imsc.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(movie_file,  codec=codec)
            
    if movie_tool=='ffmpeg-python':
        # frames may not be in order
        if movie_tool not in sys.modules:
            import ffmpeg        
        (ffmpeg.input(image_path + '/*.png', pattern_type='glob', framerate=fps)
              .output(movie_file, vcodec=codec).overwrite_output().run())

    if movie_tool=='ffmpeg':
        # frames may not be in order
        os.system('ffmpeg -framerate ' + str(fps) + '-pattern_type glob -i *.png' + codec + ' -y ' + movie_file)

def check_time(statement):
    '''
    parameters:
    -----------
    statement: string

    '''
    cProfile.run(statement)