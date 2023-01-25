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
from numpy.lib.recfunctions import append_fields

def equalize_list_sublengths(l):
    ''' Fill out each sublist-member of the list up to the length of the longest member, with redundant values (so that it can be a np.array)'''
    maxlen = max(len(v) for v in l)  
    for i in range(0,len(l)):
        l[i] += [l[i][0]] * max(maxlen - len(l[i]), 0)
    return l

class CanvasPart(object):
    '''
    superclass with subclasses CanvasComposition and CanvasArea. A CanvasPart 
    object is a defined "part" of a canvas with a set spatial border, which 
    can be any shape or size (up to the size of the full canvas) 

    attributes
    ----------
    id : str
        name used to store output. 
        When specified, 'id' is the string from the atlas file that identifies a particular composition.
    is_rectangle : bool
        True if the boundaries of the canvas part are exactly rectangular (and that there is only one boundary for the whole time). 
    pixel_changes : numpy structured array
        Pixel changes over time within the CanvasPart boundary. 
        Columns are called 'seconds', 'coord_index', 'user', 'color', 'moderator', 'in_timerange'.
        The (x,y) coordinates can be obtained with the self.pixel_changes_coords() function
    border_path : 3d numpy array with shape (number of border paths in time, number of points in path, 2)
        For each time range where a boundary is defined, contains an array of x,y coordinates defining the boundary of the CanvasPart
    border_path_times : 2d numpy array with shape (number of border paths, 2)
        The time ranges [t0,t1] in which each boundary from the border_path array is valid
    coords : 2d numpy array with shape (2, number of pixels in the CanvasPart)
        [x coordinates, y coordinates] of all the pixels contained inside the CanvasPart.
    coords_timerange : 3d numpy array with shape (number of pixels in the CanvasPart, max number of disjoint timeranges, 2)
        timeranges in which the pixels at this index (first dimension) are "on".
        Is actually a 1d numpy array (dtype=object) of list objects when there are more than 1 disjoint timeranges for a given pixel.
    xmin, xmax, ymin, ymax : integers
        limits of the smallest rectangle encompassing all pixels in boundary
    colidx_to_hex : dictionary
        dictionary with keys of color index and values of color hex code
    colidx_to_rgb : dictionary
        dictionary with keys of color index and values of color rgb
    data_path : string
        full path where the pixel data is stored
    data_file : string
        name of the pixel data file
    description : string
        description of the composition from the atlas.json. Empty in general for non-compositions.

    methods
    -------
    __init__
    __str__
    get_bounded_coords(self, show_coords =False)
    find_pixel_changes_in_boundary(self, pixel_changes_all, data_path=os.path.join(os.getcwd(),'data'))
    set_color_dictionaries(self)
    out_name(self)
    get_rgb(self, col_idx)
    set_is_rectangle(self)
    get_atlas_border(self, id, atlas=None, data_path=os.path.join(os.getcwd(), 'data'))
    reject_off_times(self)
    pixel_changes_coords(self)
    '''

    def __init__(self,
                 id='',
                 border_path=[[[]]],
                 border_path_times=[[0, var.TIME_TOTAL]],
                 atlas=None,
                 pixel_changes_all=None,
                 data_path=os.path.join(os.getcwd(), 'data'),
                 data_file='PixelChangesCondensedData_sorted.npz',
                 show_coords=False
                 ):
        '''
        Constructor for CanvasPart object

        Parameters
        ----------
        border_path, border_path_times, data_path, data_file, id : 
            directly set the attributes of the CanvasPart class, documented in the class
        pixel_changes_all : numpy recarray or None, optional
            Contains all of the pixel change data from the entire dataset. If ==None, then the whole dataset is loaded when the class instance is initialized.
        show_coords : bool, optional
            If True, plots the mask of the CanvasPart boundary
        atlas : dictionary
            Composition atlas from the atlas.json file, only needed for compositions. 
            If =None, it is extracted from the file in the get_atlas_border method.
        '''
        self.data_path = data_path
        self.data_file = data_file
        self.id = id

        # raise exceptions when CanvasPart is (or not) a composition but misses essential info
        if self.id == '' and border_path == [[[]]]:
            raise ValueError('ERROR: cannot initialise a CanvasPart which has an empty atlas id but has no user-specified border_path!')
        if self.id != '' and border_path != [[[]]]:
            raise ValueError('ERROR: cannot initialise a CanvasPart which has an atlas id but a user-specified border_path!')
        
        # get the border path from the atlas of compositions, when it is not provided
        if border_path != [[[]]]:
            border_path = equalize_list_sublengths(border_path)
            self.border_path = np.array(border_path, np.int16)
            self.border_path_times = border_path_times
            self.description = ''
        else:
            self.get_atlas_border(atlas)
        # check if path is fully inside canvas
        self.border_path[self.border_path < 0] = 0
        self.border_path[self.border_path > 1999] = 1999

        self.xmin = self.border_path[:,:,0].min()
        self.xmax = self.border_path[:,:,0].max()
        self.ymin = self.border_path[:,:,1].min()
        self.ymax = self.border_path[:,:,1].max()

        # set "is_rectangle" attribute -- True if the border_path has length 1 and is a strict rectangle
        self.set_is_rectangle()

        # set the [x, y] coords within the border
        if self.is_rectangle:
            self.coords = np.mgrid[ self.xmin:(self.xmax+1), self.ymin:(self.ymax+1) ].reshape(2,-1)
            self.coords_timerange = np.full( (len(self.coords[0]), 1, 2), self.border_path_times[0])
        else:
            self.get_bounded_coords(show_coords=show_coords)
        # reject times where a canvas quarter was off, for pixels in these quarters
        self.reject_off_times()

        # set the pixel changes within the boundary
        self.pixel_changes = None
        self.find_pixel_changes_in_boundary(pixel_changes_all)

        # set the color (hex and rgb) dictionaries
        self.colidx_to_hex = {}
        self.colidx_to_rgb = {}
        self.set_color_dictionaries()

    def __str__(self):
        return f"CanvasPart \n{'Atlas Composition, id: '+self.id if self.id!='' else 'user-defined area, name: '+self.out_name()}, \
        \n{'' if self.is_rectangle else 'not '}Rectangle, {len(self.border_path)} time-dependent border_path(s)\n{len(self.coords[0])} pixels in total, x in [{self.xmin}, {self.xmax}], y in [{self.ymin}, {self.ymax}]\
        \n{len(self.pixel_changes['seconds'])} pixel changes (including {np.count_nonzero(self.pixel_changes['in_timerange'])} in composition time ranges)\
        \n\nDescription: \n{self.description} \n\nPixel changes: \n{self.pixel_changes}\
        \n{f'    Time ranges for boundary paths: {chr(10)}{self.border_path_times}' if len(self.border_path)>1 else ''}" # chr(10) is equivalent to \n

    def out_name(self):
        if self.id != '':
            return self.id
        elif self.is_rectangle:
            return 'rectangle_'+str(self.xmin)+'.'+str(self.ymin)+'_to_'+str(self.xmax)+'.'+str(self.ymax)
        else:
            return 'area_within_'+str(self.xmin)+'.'+str(self.ymin)+'_to_'+str(self.xmax)+'.'+str(self.ymax)

    def set_is_rectangle(self):
        ''' Determines the is_rectangle attribute. Needs self.border_path to be defined.'''
        self.is_rectangle = False
        border = self.border_path
        if len(border) == 1 and len(border[0]) == 4: # only if border_path has a single 4-point-like element
            if ((border[0][0][0] == border[0][1][0] or border[0][0][1] == border[0][1][1])
            and (border[0][1][0] == border[0][2][0] or border[0][1][1] == border[0][2][1])
            and (border[0][2][0] == border[0][3][0] or border[0][2][1] == border[0][3][1])
            and (border[0][3][0] == border[0][0][0] or border[0][3][1] == border[0][0][1])
                ):
                self.is_rectangle = True

    def set_color_dictionaries(self):
        ''' Set up the dictionaries translating the color index to the corresponding rgb or hex color'''
        color_dict_file = open(os.path.join(self.data_path, 'ColorsFromIdx.json'))
        color_dict = json.load(color_dict_file)
        self.colidx_to_hex = np.empty([len(color_dict)], dtype='object')
        self.colidx_to_rgb = np.empty([len(color_dict), 3], dtype='int32')
        for (k, v) in color_dict.items():  # rather make these dictionaries numpy arrays, for more efficient use later
            self.colidx_to_hex[int(k)] = v
            self.colidx_to_rgb[int(k)] = np.asarray(ImageColor.getrgb(v))

    def get_rgb(self, col_idx):
        ''' Returns the (r,g,b) triplet corresponding to the input color index '''
        return self.colidx_to_rgb[col_idx]

    def pixel_changes_coords(self):
        ''' Returns the 2d array of the (x, y) coordinates of all pixel changes. Shape (2, number of pixel changes). '''
        return self.coords[:, self.pixel_changes['coord_index'] ]

    def get_atlas_border(self, atlas=None):
        '''
        Get the path(s) of the boundary of the composition from the atlas of the atlas.json file. 
        Also gets the time ranges in which these paths are valid, and the composition description.
        '''

        # get the atlas
        if atlas == None:
            atlas_path = os.path.join(self.data_path, 'atlas.json')
            with open(atlas_path) as f:
                atlas = json.load(f)

        # find the composition in the atlas
        for i in range(len(atlas)):
            if atlas[i]['id'] == self.id:
                id_index = i
                break

        # extract the paths and associated time ranges
        paths = atlas[id_index]['path']
        times = []
        vals = []
        for k,v in paths.items():
            if (k == 'T:0-1') and (len(paths.keys()) == 1): # if there is only one path and it has weird time tag, set it to widest timerange
                times.append([0., var.TIME_TOTAL])
                vals.append(v)
                break
            t0t1_list = k.split(',')
            for t0t1 in t0t1_list:
                if t0t1 == 'T:0-1' or t0t1 == ' T:0-1' or t0t1 == ' T' or t0t1 == 'T':
                    continue # remove the weird tag
                t0t1 = t0t1.split('-')
                t0 = t0t1[0]
                t1 = t0t1[1] if len(t0t1) > 1 else t0
                times.append([1800*(int(t0)-1), 1800*int(t1)])
                vals.append(v)

        # fill out each border_path up to the length of the longest border_path, with redundant values (so that it can be a np.array)
        vals = equalize_list_sublengths(vals)

        # fill attributes
        self.border_path = np.array(vals, dtype=np.uint16)
        self.border_path_times = np.array(times, dtype=np.float64)
        self.description = atlas[id_index]['description']

        # sort paths and times by increasing time ranges
        sort_idx = self.border_path_times[:,0].argsort() # sort according to first column
        self.border_path_times = self.border_path_times[sort_idx]
        self.border_path = self.border_path[sort_idx]


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

        coor_timerange = []
        y_coords = []
        x_coords = []

        for i in range(0, len(self.border_path)):
            # total final size of the r/place canvas
            img = np.ones((self.ymax - self.ymin + 1, self.xmax - self.xmin + 1))
            mask = np.zeros((self.ymax - self.ymin + 1, self.xmax - self.xmin + 1))

            # create mask from border_path
            cv2.fillPoly(mask, pts=np.int32( [np.add(self.border_path[i], np.array([ -int(self.xmin), -int(self.ymin)]) )] ), color=[1, 1, 1])
            masked_img = cv2.bitwise_and(img, mask)
            
            if show_coords:
                plt.figure()
                plt.imshow(masked_img)

            timerange_new = self.border_path_times[i]
            if i == 0: # keep as numpy arrays when there is only one border_path
                y_coords, x_coords = np.where(masked_img == 1)
                coor_timerange = np.full( (len(y_coords), 1, 2), timerange_new)
            else:
                if i == 1:
                    y_coords = y_coords.tolist() #switch to lists because will need to append elements
                    x_coords = x_coords.tolist()
                    coor_timerange = coor_timerange.tolist()
                for (y, x) in zip(*np.where(masked_img == 1)): # loop over the coordinates of border_path #i
                    it = 0
                    for (yref, xref) in zip(y_coords,x_coords): # loop over pre-existing coordinates
                        if y == yref and x == xref:
                            timerange_old = coor_timerange[it][-1] # last [t0, t1] timerange contains the latest times
                            if np.allclose([timerange_old[1]], [timerange_new[0]], rtol=1e-10, atol=1e-5): # are the two timeranges adjacent? 
                                coor_timerange[it][-1] = [timerange_old[0], timerange_new[1]] # merge the two timeranges
                            else:
                                coor_timerange[it].append(list(timerange_new))
                            break
                        it += 1
                    if it == len(y_coords): # when (x,y) does not exist yet in pre-existing coordinates
                        y_coords.append(y)
                        x_coords.append(x)
                        coor_timerange.append([list(timerange_new)])

        self.coords = np.vstack(( (np.array(x_coords) + self.xmin).astype(np.uint16),
                                  (np.array(y_coords) + self.ymin).astype(np.uint16)) )
        max_disjoint_timeranges = max(len(v) for v in coor_timerange)
        self.coords_timerange = np.array(coor_timerange, dtype = (np.float64 if max_disjoint_timeranges == 1 else object))

    def reject_off_times(self):
        '''
        Cut away in self.coords_timerange the times where some quarters of the canvas were off
        '''
        off1_ind = np.nonzero(np.asarray(self.coords[0]>=1000) * np.asarray(self.coords[1]<1000))[0]
        off2_ind = np.nonzero(np.asarray(self.coords[1]>=1000))[0]

        for indices, tmin in [off1_ind, var.TIME_ENLARGE1], [off2_ind, var.TIME_ENLARGE2]:
            for i in indices: # loop on indices of coordinates to be treated
                deleted_items = 0
                for j in range(0, len(self.coords_timerange[i])): # loop on different disjoint timeranges in which this pixel is active (usually 1)
                    if deleted_items > 0:
                        j -= deleted_items
                    self.coords_timerange[i][j][0] = max(self.coords_timerange[i][j][0], tmin) # remove times before tmin
                    if self.coords_timerange[i][j][1] < tmin: # if the upper time limit is below tmin, this timerange is deleted, leaving at least 1 (potentially dummy) timerange
                        if len(self.coords_timerange[i]) > 1:
                            del self.coords_timerange[i][j]
                            deleted_items += 1
                        else: 
                            self.coords_timerange[i][j][1] = tmin
                            self.coords_timerange[i][j][0] = tmin

        # switch back to numpy array if enough timeranges were removed
        if self.coords_timerange.dtype == object and max(len(v) for v in self.coords_timerange) == 1:
            self.coords_timerange = np.array(self.coords_timerange, dtype = np.float64)


    def find_pixel_changes_in_boundary(self, pixel_changes_all):
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

        # limit the pixel changes array to the min and max boundary coordinates
        ind_x = np.where((pixel_changes_all['xcoor']<=self.xmax)
                         & (pixel_changes_all['xcoor']>=self.xmin))[0]
        pixel_changes_xlim = pixel_changes_all[ind_x]
        ind_y = np.where((pixel_changes_xlim['ycoor']<=self.ymax) 
                         & (pixel_changes_xlim['ycoor']>=self.ymin))[0]
        pixel_changes_lim = pixel_changes_xlim[ind_y]
        del pixel_changes_xlim

        # find the pixel changes that correspond to pixels inside the boundary
        coords_comb = self.coords[0] + 10000.*self.coords[1]
        if not self.is_rectangle:
            # indices of pixel_changes_lim that contain the coordinates of self.coords 
            pixel_change_index = np.where(np.isin( (pixel_changes_lim['xcoor'] + 10000.*pixel_changes_lim['ycoor']), coords_comb))[0]
            pixel_changes_lim = pixel_changes_lim[pixel_change_index]

        # indices of self.coords where to find the x,y of a given pixel_change 
        coord_sort_inds = np.argsort(coords_comb)
        inds_in_sorted_coords = np.searchsorted(coords_comb[coord_sort_inds], 
                                                pixel_changes_lim['xcoor'] + 10000.*pixel_changes_lim['ycoor'])
        inds_in_coords = coord_sort_inds[inds_in_sorted_coords]

        # determine if the pixel change in the 'active' timerange for the composition 
        is_in_comp = np.full(len(pixel_changes_lim), True if self.is_rectangle else False, dtype=np.bool_)
        if not self.is_rectangle:
            s = pixel_changes_lim['seconds']
            timeranges = self.coords_timerange

            if timeranges.dtype != object: # case where the timeranges array is a numpy array, with only 1 timerange per pixel
                is_in_comp = (s[:] < timeranges[inds_in_coords[:],0,1]) & (s[:] > timeranges[inds_in_coords[:],0,0])
            
            else: # here, timeranges is not a full numpy array, so cannot use broadcasting
                for i in range(0, len(pixel_changes_lim)):
                    for timerange in timeranges[inds_in_coords[i]]:
                        is_in_comp[i] |= (s[i] > timerange[0] and s[i] < timerange[1])

        # save pixel changes as a structured array
        self.pixel_changes = np.zeros(len(pixel_changes_lim),
                                      dtype=np.dtype([('seconds', np.float64), 
                                                      ('coord_index', np.uint16 if len(self.coords[0]) < 65530 else np.uint32 ), 
                                                      ('user', np.uint32), 
                                                      ('color', np.uint8), 
                                                      ('in_timerange', np.bool_),
                                                      ('moderator', np.bool_)]) )
        self.pixel_changes['seconds'] = np.array(pixel_changes_lim['seconds'])
        self.pixel_changes['coord_index'] = np.array(inds_in_coords)
        self.pixel_changes['user'] = np.array(pixel_changes_lim['user'])
        self.pixel_changes['color'] = np.array(pixel_changes_lim['color'])
        self.pixel_changes['in_timerange'] = np.array(is_in_comp)
        self.pixel_changes['moderator'] = np.array(pixel_changes_lim['moderator'])

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
    times : 1d array of floats
        time limits of intervals in which to plot (in seconds)
    delete_bmp / delete_png : boolean, optional
        if True, the .bmp / .png files are deleted after their size is determined
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
    coords = canvas_part.pixel_changes_coords()
    color = np.array(canvas_part.pixel_changes['color'])

    num_time_steps = len(times)-1
    file_size_bmp = np.zeros(num_time_steps+1)
    file_size_png = np.zeros(num_time_steps+1)

    pixels = np.full((canvas_part.ymax - canvas_part.ymin + 1, canvas_part.xmax - canvas_part.xmin + 1, 3), 255, dtype=np.uint8) #[r,g,b] will be readable as (r,g,b) ##### fill as white first ##### the pixels must be [y,x,rgb]
        
    out_path = os.path.join(os.getcwd(), 'figs', 'history_' + canvas_part.out_name())
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

    for t_step_idx in range(0, num_time_steps+1):  # fixed this: want a blank image at t=0
        if print_progress:
            if t_step_idx/num_time_steps > i_fraction_print/10:
                i_fraction_print += 1
                print('Ran', 100*t_step_idx/num_time_steps, '%% of the steps')

        # get the indices of the times within the interval
        t_inds = np.where((seconds >= times[t_step_idx - 1]) & (seconds < times[t_step_idx]))[0]
        t_inds_list.append(t_inds)
        if len(t_inds) != 0:
            pixels[coords[1, t_inds]-canvas_part.ymin, coords[0, t_inds]-canvas_part.xmin, :] = canvas_part.get_rgb(color[t_inds])

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


def plot_compression(file_size_bmp, file_size_png, times, out_name=''):
    '''
    plot the file size ratio over time

    parameters
    ----------
    file_size_bmp : float
        size of png image in bytes
    file_size_png : float
        size of png image in bytes
    times : 1d array of floats
        time intervals at which to plot (in seconds)
    out_name : string
        for the naming of the output saved plot
    '''

    plt.figure()
    plt.plot(times, file_size_png/file_size_bmp)
    sns.despine()
    plt.ylabel('Computable Information Density (file size ratio)')
    plt.xlabel('Time (s)')
    plt.savefig(os.path.join(os.getcwd(), 'figs', 'history_' + out_name, '_file_size_compression_ratio.png'))


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
    pixel_changes_all : numpy structured array
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
                                times,
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
                    times,
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
