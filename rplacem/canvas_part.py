import cProfile
import glob
import json
import math
import os
import pickle
import shutil
import sys
from operator import mod
from re import T
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageColor
import Variables.Variables as var


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
    pixel_changes : numpy recarray
        Pixel changes over time within the CanvasPart boundary. 
        Columns are called 'seconds', 'xcoor', 'ycoor', 'user', 'color'
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
    find_pixel_changes_in_boundary(self, 
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
                 data_path=os.path.join(os.getcwd(), 'data'),
                 data_file='PixelChangesCondensedData_sorted.npz',
                 show_coords=False):
        '''
        Constructor for CanvasPart object

        Parameters
        ----------
        border_path : 2d numpy array with shape (number of points in path, 2)
            Array of x,y coordinates defining the boundary of the CanvasPart
        pixel_changes_all : numpy recarray or None, optional
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
        self.find_pixel_changes_in_boundary(pixel_changes_all)

    def set_color_dictionaries(self):
        ''' set the color dictionaries from the data file '''

        color_dict_file = open(os.path.join(self.data_path, 'ColorsFromIdx.json'))
        color_dict = json.load(color_dict_file)
        self.colidx_to_hex = np.empty([len(color_dict)], dtype='object')
        self.colidx_to_rgb = np.empty([len(color_dict), 3], dtype='int32')
        for (k, v) in color_dict.items():  # rather make these dictionaries numpy arrays, for more efficient use later
            self.colidx_to_hex[int(k)] = v
            self.colidx_to_rgb[int(k)] = np.asarray(ImageColor.getrgb(v))

    def get_rgb(self, col_idx):
        ''' Get (r,g,b) triplet from the color index '''
        return self.colidx_to_rgb[col_idx]

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
        img = np.ones((2000, 2000))
        mask = np.zeros((2000, 2000))

        # create mask from border_path
        cv2.fillPoly(mask, pts=[self.border_path], color=[1, 1, 1])
        masked_img = cv2.bitwise_and(img, mask)

        if show_coords:
            plt.figure()
            plt.imshow(masked_img)

        y_coords_boundary, x_coords_boundary = np.where(masked_img == 1)

        self.y_coords = y_coords_boundary.astype(np.uint16)
        self.x_coords = x_coords_boundary.astype(np.uint16)

    def find_pixel_changes_in_boundary(self,
                                    pixel_changes_all
                                    ):
        '''
        Find all the pixel changes within the boundary of the CanvasPart object
        and set the pixel_changes attribute of the CanvasPart object accordingly

        parameters
        ----------
        pixel_changes_all : numpy recarray or None
            Contains all of the pixel change data from the entire dataset
        data_path : str, optional
            Path to where the pixel data file is stored
        '''
        if pixel_changes_all is None:
            pixel_changes_all = get_all_pixel_changes()

        x_max = np.max(self.x_coords)
        x_min = np.min(self.x_coords)
        y_max = np.max(self.y_coords)
        y_min = np.min(self.y_coords)

        # limit the pixel changes array to the min and max boundary coordinates
        ind_x = np.where((pixel_changes_all['xcoor']<=x_max) 
                          & (pixel_changes_all['xcoor']>=x_min))[0]
        pixel_changes_xlim = pixel_changes_all[ind_x]
        ind_y = np.where((pixel_changes_xlim['ycoor']<=y_max) 
                         & (pixel_changes_xlim['ycoor']>=y_min))[0]
        pixel_changes_lim = pixel_changes_xlim[ind_y]

        # find the pixel changes that correspond to pixels inside the boundary
        pixel_change_index = np.where(np.isin((pixel_changes_lim['xcoor'] + 10000.*pixel_changes_lim['ycoor']), 
                                       (self.x_coords + 10000.*self.y_coords)))[0]
        
        self.pixel_changes = pixel_changes_lim[pixel_change_index] 


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
    find_pixel_changes_in_boundary(self, 
                                pixel_changes_all, 
                                data_path=os.path.join(os.getcwd(),'data'))
        Inherited from superclass

    '''

    def __init__(self,
                 id,
                 pixel_changes_all=None,
                 atlas=None,
                 data_path=os.path.join(os.getcwd(), 'data'),
                 data_file='PixelChangesCondensedData_sorted.npz',
                 show_coords=False):
        '''
        Constructor for CanvasComposition object

        Parameters
        ----------
        id: string
            The string from the atlas file that identifies a particular composition
        pixel_changes_all : numpy recarray or None, optional
            Contains all of the pixel change data from the entire dataset
        data_path : str, optional
            Path to where the pixel data file is stored
        show_coords : bool, optional
            If True, plots the mask of the CanvasPart boundary
        '''

        # get the border_path from the atlas.json file
        # path0 is the initial path.
        # TODO: add handling for if path changes over time
        paths, path0 = self.get_atlas_border(id, atlas=atlas, data_path=data_path)
        self.id = id 
 
        super().__init__(path0,
                         pixel_changes_all=pixel_changes_all,
                         data_path=data_path,
                         data_file=data_file,
                         show_coords=show_coords)

    def get_atlas_border(self, id, atlas=None, data_path=os.path.join(os.getcwd(), 'data')):
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
        if atlas == None:
            atlas_path = os.path.join(data_path, 'atlas.json')
            composition_classification_file = open(atlas_path)
            atlas = json.load(composition_classification_file)

        for i in range(len(atlas)):
            if atlas[i]['id'] == id:
                id_index = i
                break

        paths = atlas[id_index]['path']
        path0 = np.array(list(paths.values())[0])  # initial path

        self.id_name = str(atlas[id_index]['name'])
        if atlas == None:
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
    find_pixel_changes_in_boundary(self, 
                                pixel_changes_all, 
                                data_path=os.path.join(os.getcwd(),'data'))
        Inherited from superclass

    '''

    def __init__(self,
                 border_path,
                 pixel_changes_all,
                 data_path=os.path.join(os.getcwd(), 'data'),
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
    pixel_changes : numpy recarray
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
                        times, # in seconds
                        part_name='cp',  # only for name of output
                        delete_bmp=True,
                        delete_png=False,
                        show_plot=True,
                        print_progress=True
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
    xcoor = np.array(canvas_part.pixel_changes['xcoor'])
    ycoor = np.array(canvas_part.pixel_changes['ycoor'])
    color = np.array(canvas_part.pixel_changes['color'])

    num_time_steps = len(times)
    file_size_bmp = np.zeros(num_time_steps+1)
    file_size_png = np.zeros(num_time_steps+1)

    path_coords = canvas_part.border_path
    x_min = np.min(path_coords[:, 0])
    x_max = np.max(path_coords[:, 0])
    y_min = np.min(path_coords[:, 1])
    y_max = np.max(path_coords[:, 1])

    pixels = np.full((y_max - y_min + 1, x_max - x_min + 1, 3), 255, dtype=np.uint8)  # consider that [r,g,b] will be readable as (r,g,b) ##### fill as white first ##### not sure why, but the pixels should be [y,x,rgb]
    out_path = os.path.join(os.getcwd(), 'figs', 'history_' + part_name)
    out_path_time = os.path.join(out_path, 'VsTime')
    try:
        os.makedirs(out_path)
    except OSError:  # empty directory if it already exists
        shutil.rmtree(out_path_time)
        #os.makedirs(out_path)
    os.makedirs(os.path.join(out_path_time))

    if show_plot:
        ncols = np.min([num_time_steps, 10])
        nrows = np.max([1, int(math.ceil(num_time_steps/10))])
        # fig = plt.figure(figsize=(10,nrows)) # height corresponds in inches to number of rows. auto dpi is 100
        #gs = fig.add_gridspec(nrows, ncols, hspace=0.05, wspace=0.05)
        fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        #ax = gs.subplots(sharex=True, sharey=True)
        rowcount = 0
        colcount = 0
    t_inds_list = []

    i_fraction_print = 0  # only for output of a message when a fraction of the steps are ran

    for t_step_idx in range(1, num_time_steps):  # fixed this: want a blank image at t=0
        if print_progress:
            if t_step_idx/num_time_steps > i_fraction_print/10:
                i_fraction_print += 1
                print('Ran', 100*t_step_idx/num_time_steps, '% of the steps')

        # get the indices of the times within the interval
        t_inds = np.where((seconds >= times[t_step_idx - 1]) & (seconds < times[t_step_idx]))[0]
        t_inds_list.append(t_inds)
        if len(t_inds) != 0:
            pixels[ycoor[t_inds]-y_min, xcoor[t_inds]-x_min, :] = canvas_part.get_rgb(color[t_inds])

        # save image
        im = Image.fromarray(pixels)
        im_path = os.path.join(out_path_time, 'canvaspart_time{:06d}'.format(int(times[t_step_idx])))
        im.save(im_path + '.png')
        im.save(im_path + '.bmp')
        file_size_png[t_step_idx] = get_file_size(im_path + '.png')
        file_size_bmp[t_step_idx] = get_file_size(im_path + '.bmp')
        if delete_bmp:
            os.remove(im_path + '.bmp')
        if delete_png:
            os.remove(im_path + '.png')

        if show_plot:
            if t_step_idx>-1:
                if len(ax.shape) == 2:
                    ax_single = ax[rowcount, colcount]
                else:
                    ax_single = ax[t_step_idx]
                ax_single.axis('off')
                show_canvas_part(pixels, ax=ax_single)

                if colcount < 9:
                    colcount += 1
                else:
                    colcount = 0
                    rowcount += 1
    if print_progress:
        print('produced', num_time_steps, 'images vs time')
    return file_size_bmp, file_size_png, t_inds_list


def plot_compression(file_size_bmp, file_size_png, time_interval, total_time, part_name=''):
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
    plt.savefig(os.path.join(os.getcwd(), 'figs', 'history_' + part_name, '_file_size_compression_ratio.png'))


def get_all_pixel_changes(data_file='PixelChangesCondensedData_sorted.npz',
                          data_path=os.path.join(os.getcwd(), 'data')):
    '''
    load all the pixel change data and put it in a numpy array for easy access

    parameters
    ----------
    data_file : string that ends in .npz, optional
    data_path : location of data, assumed to be in cwd in 'data' folder, optional

    returns
    -------
    pixel_changes_all : numpy recarray
            Contains all of the pixel change data from the entire dataset
    '''
    pixel_changes_all_npz = np.load(os.path.join(data_path, data_file))

    # save pixel changes as a structured array
    pixel_changes_all = np.zeros(len(pixel_changes_all_npz['seconds']),
                                    dtype=np.dtype([('seconds', np.float64), 
                                                    ('xcoor', np.uint16), 
                                                    ('ycoor', np.uint16), 
                                                    ('user', np.uint32), 
                                                    ('color', np.uint8), 
                                                    ('moderator', np.bool_)])
                                                  )
    pixel_changes_all['seconds'] = np.array(pixel_changes_all_npz['seconds'])
    pixel_changes_all['xcoor'] = np.array(pixel_changes_all_npz['pixelXpos'])
    pixel_changes_all['ycoor'] = np.array(pixel_changes_all_npz['pixelYpos'])
    pixel_changes_all['user'] = np.array(pixel_changes_all_npz['userIndex'])
    pixel_changes_all['color'] = np.array(pixel_changes_all_npz['colorIndex'])
    pixel_changes_all['moderator'] = np.array(pixel_changes_all_npz['moderatorEvent'])

    return pixel_changes_all


def save_canvas_part_time_steps(canvas_comp,
                                time_inds_list_comp,
                                time_interval,
                                file_size_bmp,
                                file_size_png,
                                data_path=os.path.join(os.getcwd(), 'data'),
                                file_name='canvas_part_data'):
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


def load_canvas_part_time_steps(data_path=os.path.join(os.getcwd(), 'data'),
                                file_name='canvas_part_data'):
    '''
    save the variables associated with the CanvasPart object 
    '''
    file_path = os.path.join(data_path, file_name + '.pickle')
    with open(file_path, 'rb') as f:
        canvas_part_parameters = pickle.load(f)

    return canvas_part_parameters


def save_movie(image_path,
               fps=1,
               movie_tool='moviepy',
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
    image_files = list(np.sort(glob.glob(os.path.join(image_path, '*.png'))))
    png_name0 = os.path.basename(image_files[0][0:-15])
    movie_name = png_name0 + '_fps' + str(fps)
    movie_file = os.path.join(image_path, movie_name) + '.' + video_type

    if movie_tool == 'moviepy':
        if 'imsc' not in sys.modules:
            import moviepy.video.io.ImageSequenceClip as imsc
        clip = imsc.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(movie_file,  codec=codec)

    if movie_tool == 'ffmpeg-python':
        # frames may not be in order
        if movie_tool not in sys.modules:
            import ffmpeg
        (ffmpeg.input(image_path + '/*.png', pattern_type='glob', framerate=fps)
         .output(movie_file, vcodec=codec).overwrite_output().run())

    if movie_tool == 'ffmpeg':
        # frames may not be in order
        os.system('ffmpeg -framerate ' + str(fps) + '-pattern_type glob -i *.png' + codec + ' -y ' + movie_file)


def check_time(statement, sort = 'cumtime'):
    '''
    parameters:
    -----------
    statement: string

    '''
    cProfile.run(statement, sort=sort)
