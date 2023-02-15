import json
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
import rplacem.variables_rplace2022 as var
import rplacem.utilities as util

class CanvasPart(object):
    '''
    A CanvasPart object is a defined "part" of a canvas with a set spatial border, which 
    can be any shape or size (up to the size of the full canvas)

    attributes
    ----------
    id : str
        When specified, 'id' is the string from the atlas file that identifies a particular composition.
            Set in __init__()
    is_rectangle : bool
        True if the boundaries of the canvas part are exactly rectangular (and that there is only one boundary for the whole time). 
            Set in _set_is_rectangle()
    pixel_changes : numpy structured array
        Pixel changes over time within the CanvasPart boundary. 
        Columns are called 'seconds', 'coord_index', 'user', 'color', 'moderator', 'active'.
        The (x,y) coordinates can be obtained with the self.pixchanges_coords() function
            Set in_find_pixel_changes_in_boundary()
    border_path : 3d numpy array with shape (number of border paths in time, number of points in path, 2)
        For each time range where a boundary is defined, contains an array of x,y coordinates defining the boundary of the CanvasPart
            Set in __init__()
    border_path_times : 2d numpy array with shape (number of border paths, 2)
        The time ranges [t0,t1] in which each boundary from the border_path array is valid
            Set in __init__()
    coords : 2d numpy array with shape (2, number of pixels in the CanvasPart)
        [x coordinates, y coordinates] of all the pixels contained inside the CanvasPart.
            Set in __init__() (if is_rectangle) or in _get_bounded_coords().
    coords_timerange : 3d numpy array with shape (number of pixels in the CanvasPart, max number of disjoint timeranges, 2)
        [pixel index, index of time ranges, start time and end time]
        timeranges in which the pixels at this index (first dimension) are "on".
        Is actually a 1d numpy array (dtype=object) of list objects when there are more than 1 disjoint timeranges for a given pixel.
            Set in __init__() (if is_rectangle) or in _get_bounded_coords().
    xmin, xmax, ymin, ymax : integers
        limits of the smallest rectangle encompassing all pixels in boundary. 
            Set in __init__()
    description : string
        description of the composition from the atlas.json. Empty in general for non-compositions. 
            Set in _get_atlas_border()
    quarter1_coordinds, quarter2_coordinds, quarter34_coordinds:
        the indices (referring to the self.coords array) that point to coordinates
        in the 1st, 2nd, or 3rd and 4th quarters (which were not all "on" from the beginning).
            Set in set_quarters_coord_inds(), at the responsibility of the user.
    has_loc_jump : bool
        True if the canvas part 'jumps' from one location to another. If at least one pixel coordinate is active 
        during the time that the cavnas_part is active, then there is no jump and has_loc_jump is set to False. This condition
        allows for small shifts in the canvas location, which do not constitute a jump. 
            Set in _set_has_loc_jump()
    pixch_sortcoord : 1d numpy array of length (number of pixel changes)
        indices of pixel changes (from argsort) that sort the pixel changes in terms of 'coord_index'
            Set in _find_redundant_pix_change(), within _find_pixel_changes_in_boundary()
    pixch_sortuser : 1d numpy array of length (number of pixel changes)
        indices of pixel changes (from argsort) that sort the pixel changes in terms of 'user'
            Set in _find_cheated_pix_change(), within _find_pixel_changes_in_boundary()
    

    methods
    -------
    private:
        __init__
        __str__
    protected:
        _set_is_rectangle(self)
        _get_atlas_border(self, id, atlas=None)
        _reject_off_times(self)
        _get_bounded_coords(self, show_coords=False)
        _find_pixel_changes_in_boundary(self, pixel_changes_all, verbose)
        _set_has_loc_jump(self)
        _find_redundant_pix_change(self)
        _find_cheated_pix_change(self)
    public:
        out_name(self)
        pixchanges_coords(self)
        pixchanges_coords_offset(self)
        coords_offset(self)
        width(self, xory=-1)
        num_pix(self)
        white_image(self, dimension=2, images_number=-1)
        set_quarters_coord_inds(self)
        intimerange_pixchanges_inds(self, t0, t1)
        active_coord_inds(self, t0, t1)
        select_active_pixchanges_inds(self, input_inds)
        save_object(self)
    '''


    # PRIVATE METHODS

    def __init__(self,
                 id='',
                 border_path=[[[]]],
                 border_path_times=np.array([[0, var.TIME_TOTAL]]),
                 atlas=None,
                 pixel_changes_all=None,
                 show_coords=False,
                 verbose=False,
                 save=False
                 ):
        '''
        Constructor for CanvasPart object

        Parameters
        ----------
        id, border_path, border_path_times: 
            directly set the attributes of the CanvasPart class, documented in the class
        atlas : dictionary
            Composition atlas from the atlas.json file, only needed for compositions. 
            If =None, it is extracted from the file in the _get_atlas_border method.
        pixel_changes_all : numpy recarray or None, optional
            Contains all of the pixel change data from the entire dataset. If ==None, then the whole dataset is loaded when the class instance is initialized.
        show_coords : bool, optional
            If True, plots the mask of the CanvasPart boundary
        verbose : bool
            Whether to print progress
        save : bool 
            Whether to save the final canvas part in a pickle file
        '''
        self.id = id

        # raise exceptions when CanvasPart is (or not) a composition but misses essential info
        if self.id == '' and border_path == [[[]]]:
            raise ValueError('ERROR: cannot initialise a CanvasPart which has an empty atlas id but has no user-specified border_path!')
        if self.id != '' and border_path != [[[]]]:
            raise ValueError('ERROR: cannot initialise a CanvasPart which has an atlas id but a user-specified border_path!')
        
        # get the border path from the atlas of compositions, when it is not provided
        if verbose:
            print('set border path (with _get_atlas_border() if composition)')
        if border_path != [[[]]]:
            border_path = util.equalize_list_sublengths(border_path)
            self.border_path = np.array(border_path, np.int16)
            self.border_path_times = border_path_times
            self.description = ''
        else:
            self._get_atlas_border(atlas)
        # check if path is fully inside canvas
        self.border_path[self.border_path < 0] = 0
        self.border_path[self.border_path > 1999] = 1999

        self.xmin = self.border_path[:,:,0].min()
        self.xmax = self.border_path[:,:,0].max()
        self.ymin = self.border_path[:,:,1].min()
        self.ymax = self.border_path[:,:,1].max()

        # set "is_rectangle" attribute -- True if the border_path has length 1 and is a strict rectangle
        self._set_is_rectangle()

        # set the [x, y] coords within the border
        if verbose:
            print('set coordinates (with _get_bounded_coords() if is not rectangle)')
        if self.is_rectangle:
            self.coords = np.mgrid[ self.xmin:(self.xmax+1), self.ymin:(self.ymax+1) ].reshape(2,-1)
            self.coords_timerange = np.full( (len(self.coords[0]), 1, 2), self.border_path_times[0])
        else:
            self._get_bounded_coords(show_coords=show_coords)
        # reject times where a canvas quarter was off, for pixels in these quarters
        if verbose:
            print('_reject_off_times()')
        self._reject_off_times()

        # set the pixel changes within the boundary
        if verbose:
            print('set pixel changes with _find_pixel_changes_in_boundary()')
        self.pixel_changes = None
        self._find_pixel_changes_in_boundary(pixel_changes_all, verbose)

        # set the boolean describing whether the canvas_part jumps to a new part of the canvas
        if verbose:
            print('_set_has_loc_jump()')
        self._set_has_loc_jump()

        if save:
            self.save_object()

    def __str__(self):
        return f"CanvasPart \n{'Atlas Composition, id: '+self.id if self.id!='' else 'user-defined area, name: '+self.out_name()}, \
        \n{'' if self.is_rectangle else 'not '}Rectangle, {len(self.border_path)} time-dependent border_path(s)\n{len(self.coords[0])} pixels in total, x in [{self.xmin}, {self.xmax}], y in [{self.ymin}, {self.ymax}]\
        \n{len(self.pixel_changes['seconds'])} pixel changes (including {np.count_nonzero(self.pixel_changes['active'])} in composition time ranges)\
        \n\nDescription: \n{self.description} \n\nPixel changes: \n{self.pixel_changes}\
        \n{f'    Time ranges for boundary paths: {chr(10)}{self.border_path_times}' if len(self.border_path)>1 else ''}" # chr(10) is equivalent to \n


    # PROTECTED METHODS

    def _set_is_rectangle(self):
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

    def _get_atlas_border(self, atlas=None):
        '''
        Get the path(s) of the boundary of the composition from the atlas of the atlas.json file. 
        Also gets the time ranges in which these paths are valid, and the composition description.
        '''

        # get the atlas
        if atlas == None:
            atlas_path = os.path.join(var.DATA_PATH, 'atlas.json')
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
        vals = util.equalize_list_sublengths(vals)

        # fill attributes
        self.border_path = np.array(vals, dtype=np.int16)
        self.border_path_times = np.array(times, dtype=np.float64)
        self.description = atlas[id_index]['description']

        # sort paths and times by increasing time ranges
        sort_idx = self.border_path_times[:,0].argsort() # sort according to first column
        self.border_path_times = self.border_path_times[sort_idx]
        self.border_path = self.border_path[sort_idx]

    def _get_bounded_coords(self, show_coords=False):
        '''
        Finds the x and y coordinates within the boundary (including the time ranges) of the CanvasPart object. 
        Sets the coords and coords_timerange attributes of the CanvasPart object.

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
            img = np.ones((self.width(1), self.width(0)))
            mask = np.zeros((self.width(1), self.width(0)))

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
                    for (yref, xref) in zip(y_coords, x_coords): # loop over pre-existing coordinates
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

        self.coords = np.vstack(( (np.array(x_coords) + self.xmin).astype(np.int16),
                                  (np.array(y_coords) + self.ymin).astype(np.int16)) )
        max_disjoint_timeranges = max(len(v) for v in coor_timerange)
        self.coords_timerange = np.array(coor_timerange, dtype = (np.float64 if max_disjoint_timeranges == 1 else object))

    def _reject_off_times(self):
        '''
        Cut away in self.coords_timerange the times at which some quarters of the canvas were off
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


    def _find_pixel_changes_in_boundary(self, pixel_changes_all, verbose):
        '''
        Find all the pixel changes within the boundary of the CanvasPart object
        and set the pixel_changes attribute of the CanvasPart object accordingly

        parameters
        ----------
        pixel_changes_all : numpy recarray or None
            Contains all of the pixel change data from the entire dataset
        '''
        if pixel_changes_all is None:
            pixel_changes_all = util.get_all_pixel_changes()
        
        if verbose:
            print('    find pixels inside boundary')        
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
        if verbose:
            print('    sort pixel_changes vs coordinates')   
        coord_sort_inds = np.argsort(coords_comb)
        inds_in_sorted_coords = np.searchsorted(coords_comb[coord_sort_inds], 
                                                pixel_changes_lim['xcoor'] + 10000.*pixel_changes_lim['ycoor'])
        inds_in_coords = coord_sort_inds[inds_in_sorted_coords]

        # determine if the pixel change is in the 'active' timerange for the composition 
        if verbose:
            print('    determine if the pixel change is in the active timerange')        
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
        if verbose:
            print('    make pixel_changes output')
        self.pixel_changes = np.zeros(len(pixel_changes_lim),
                                      dtype=np.dtype([('seconds', np.float64), 
                                                      ('coord_index', np.uint16 if len(self.coords[0]) < 65530 else np.uint32 ), 
                                                      ('user', np.uint32), 
                                                      ('color', np.uint8), 
                                                      ('active', np.bool_),
                                                      ('moderator', np.bool_),
                                                      ('cooldown_cheat', np.bool_),
                                                      ('redundant_col', np.bool_),
                                                      ('redundant_colanduser', np.bool_)]) )
        self.pixel_changes['seconds'] = np.array(pixel_changes_lim['seconds'])
        self.pixel_changes['coord_index'] = np.array(inds_in_coords)
        self.pixel_changes['user'] = np.array(pixel_changes_lim['user'])
        self.pixel_changes['color'] = np.array(pixel_changes_lim['color'])
        self.pixel_changes['active'] = np.array(is_in_comp)
        self.pixel_changes['moderator'] = np.array(pixel_changes_lim['moderator'])

        # find redundant pixel changes
        if verbose:
            print('    find redundant pixel changes')
        (self.pixel_changes['redundant_col'], self.pixel_changes['redundant_colanduser']) = self._find_redundant_pix_change()

        # find changes with cheated cooldown (<250 seconds)
        if verbose:
            print('    find changes with cheated cooldown')
        self.pixel_changes['cooldown_cheat'] = self._find_cheated_pix_change()


    def _find_cheated_pix_change(self):
        ''' 
        Finds all the pixel_changes that cheated the 300 seconds cooldown significantly,
        meaning all changes done by a user who had done a pixel change less than 250s earlier.
        '''
        # sorting vs user index, then the time ordering should be conserved at 2nd level
        sorting = self.pixel_changes.argsort(order=['user']) 
        self.pixch_sortuser = sorting
        
        # exclude moderator events
        pix_change_cheat = np.zeros(len(self.pixel_changes))
        pix_change_cheat[ self.pixel_changes['moderator'] ] = 1

        ind_nomod = np.where( np.logical_not(self.pixel_changes['moderator'][sorting]) )[0]
        pixchanges = self.pixel_changes[sorting[ind_nomod]]

        # get the time difference between pixel changes (n+1) and n for same user, and when the user changes
        timedif = np.hstack((1e5, np.diff(pixchanges['seconds'])))
        userchange = np.hstack((1e5, np.diff(pixchanges['user'])))
        ind_cheat = np.where((timedif < var.COOLDOWN_MIN) & (userchange == 0))[0]

        pix_change_cheat[sorting[ind_nomod[ind_cheat]]] = 1
        return pix_change_cheat.astype(bool)

    def _find_redundant_pix_change(self):
        ''' 
        Finds all the pixel_changes that are redundant and returns
        a boolean array indicating pixel changes whose color is redundant.

        The second returned array is of the same form, 
        but says if the color AND the user is the same.
        '''
        # order pixel changes versus the coordinates (then the time ordering should be conserved at second level)
        sorting = np.argsort(self.pixel_changes, order = ['coord_index'])
        self.pixch_sortcoord = sorting
        color = self.pixel_changes['color'][sorting]
        coord = self.pixel_changes['coord_index'][sorting]
        user = self.pixel_changes['user'][sorting]

        # look for changes of the same color and concerning the same pixel
        redundant_ind = np.where((np.diff(color) == 0) & (np.diff(coord) == 0))[0] + 1 
        ind_xtra_redundant = np.where(user[redundant_ind] == user[redundant_ind-1])[0]

        # returned objects
        pix_change_redundant = np.zeros(len(color))
        pix_change_redundant[sorting[redundant_ind]] = 1
        pix_change_extraredundant = np.zeros(len(color))
        pix_change_extraredundant[sorting[redundant_ind[ind_xtra_redundant]]] = 1
        return (pix_change_redundant.astype(bool), pix_change_extraredundant.astype(bool))

    def _set_has_loc_jump(self):
        '''
        Finds wether the canvas_part has a location jump
        and sets the has_loc_jump boolean attribute accordingly

        A location jump is defined as a change in the border path
        where no pixel contained in the border is the same before and 
        after. 
        '''
        border_times = np.ndarray.flatten(self.border_path_times)
        # get full time range during which compo was active
        _, times_unique_inv = np.unique(border_times, return_inverse=True)
        time_reps = np.where(np.diff(times_unique_inv) == 0)[0]
        time_reps_tot = np.hstack([time_reps,time_reps + 1])

        comp_active_time = []
        for i in range(len(border_times)):
            if ~np.isin(border_times[i], border_times[time_reps_tot]):
                comp_active_time.append(border_times[i])         
        comp_active_time = np.reshape(np.array(comp_active_time, dtype = np.float64), (-1,2))
        
        time_ranges_pix = np.array(self.coords_timerange)
        self.has_loc_jump = True
        # run over all pixels until finding one that is active the whole time
        for i in range(len(time_ranges_pix)):
            timerange = np.array(time_ranges_pix[i], dtype = np.float64)
            if len(comp_active_time) != len(timerange):
                cont_pix_bool = False
            else: # is the timerange for this pixel equal to the active timerange of the whole compo?
                cont_pix_bool = np.allclose(timerange, comp_active_time, rtol=1e-10, atol=1e-5)
            if cont_pix_bool:
                self.has_loc_jump = False
                break

    # PUBLIC METHODS

    def active_coord_inds(self, t0, t1):
        ''' Return indices of coordinates for which the interval [t0, t1] intersects with the 'active' timerange '''
        num_coord = self.coords.shape[1]
        timeranges = self.coords_timerange

        if self.is_rectangle:
            (t0ref, t1ref) = timeranges[0][0]
            if (t0 >= t0ref or t1 > t0ref) and (t0 < t1ref or t1 <= t1ref):
                inds_active = np.arange(0, num_coord)
            else:
                inds_active = np.array([])

        else:
            if timeranges.dtype != object: # case where the timeranges array is a numpy array, with only 1 timerange per pixel
                # tried to reduce this through distributivity of logical operations, but inconclusive
                inds_active = np.where( ((t0 >= timeranges[:,0,0]) | (t1 > timeranges[:,0,0])) 
                                            & ((t0 < timeranges[:,0,1]) | (t1 <= timeranges[:,0,1])) )
            else: # dirty python here: timeranges is not a full numpy array, so cannot use broadcasting
                inds_active = []
                for i in range(0, num_coord):
                    active = False
                    for timerange in timeranges[i]:
                        active |= (((t0 >= timerange[0]) | (t1 > timerange[0])) 
                                        & ((t0 < timerange[1]) | (t1 <= timerange[1])) )
                    if active:
                        inds_active.append(i)
                inds_active = np.array(inds_active, dtype=int)

        return inds_active

    def intimerange_pixchanges_inds(self, t0, t1):
        ''' Indices in self.pixel_changes whose time is in the range [t0, t1['''
        seconds = np.array(self.pixel_changes['seconds'])
        return np.where((seconds >= t0) & (seconds < t1))[0]

    def select_active_pixchanges_inds(self, input_inds):
        ''' Returns the part of the input_inds indices (referring to self.pixel_changes)
        that point to an active pixel change (meaning its time is within border_path_times)'''
        active = np.array(self.pixel_changes['active'])
        return input_inds[ np.where(active[input_inds])[0] ]

    def out_name(self):
        ''' Returns standard name used to identify the composition (used in output paths).
        Is unique, except for user-defined border_path areas that are not rectangles.'''
        if self.id != '':
            return self.id
        elif self.is_rectangle:
            return 'rectangle_'+str(self.xmin)+'.'+str(self.ymin)+'_to_'+str(self.xmax)+'.'+str(self.ymax)
        else:
            return 'area_within_'+str(self.xmin)+'.'+str(self.ymin)+'_to_'+str(self.xmax)+'.'+str(self.ymax)

    def pixchanges_coords(self):
        ''' Returns the 2d array of the (x, y) coordinates of all pixel changes. 
        Shape (2, number of pixel changes) '''
        return self.coords[:, self.pixel_changes['coord_index'] ]

    def pixchanges_coords_offset(self):
        ''' Returns the (x, y) coordinates of all pixel changes, but starting from x=0 and y=0. 
        Shape (2, number of pixel changes) '''
        res = self.pixchanges_coords()
        res[0] -= self.xmin
        res[1] -= self.ymin
        return res

    def coords_offset(self):
        ''' Returns the (x, y) coordinates of the canvas part, but starting from x=0 and y=0. 
        Shape (2, number of pixels) '''
        res = np.copy(self.coords)
        res[0] -= self.xmin
        res[1] -= self.ymin
        return res

    def width(self, xory=-1):
        ''' widest size of the canvas part in the x and y dimensions '''
        if xory == 0:
            return self.xmax - self.xmin + 1
        if xory == 1:
            return self.ymax - self.ymin + 1
        else:
            return np.array([self.xmax - self.xmin + 1, self.ymax - self.ymin + 1])
    
    def num_pix(self):
        ''' number of pixels of the canvas part that are active at some time '''
        return self.coords.shape[1]

    def white_image(self, dimension=2, images_number=-1):
        ''' Creates a white image of the size of the canvas part. 
        It can be a 1D array of the length of the coordinates;
        or a 2D array being the actual pixels/image of the part; 
        or a 3D array containing [images_number] of these 2D images. '''
        if dimension == 1:
            return np.full(self.num_pix(), var.WHITE, dtype='int8')
        if dimension == 2:
            return np.full((self.width(1), self.width(0)), var.WHITE, dtype=np.int8)
        if dimension == 3:
            return np.full((images_number, self.width(1), self.width(0)), var.WHITE, dtype=np.int8)

    def set_quarters_coord_inds(self):
        ''' Set the attributes quarter1_coordinds, quarter2_coordinds, quarter34_coordinds '''
        self.quarter1_coordinds = np.where((self.coords[0]<1000) & (self.coords[1]<1000))
        self.quarter2_coordinds = np.where((self.coords[0]>=1000) & (self.coords[1]<1000))
        self.quarter34_coordinds = np.where(self.coords[1]>=1000)

    def save_object(self):
        '''
        save the whole CanvasPart object in a pickle file
        '''
        file_path = os.path.join(var.DATA_PATH, 'CanvasPart_'+ self.out_name() + '.pickle')
        with open(file_path, 'wb') as handle:
            pickle.dump(self,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
 

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


def save_canvas_part_time_steps(canvas_comp,
                                time_inds_list_comp,
                                times,
                                file_size_bmp,
                                file_size_png,
                                file_name='canvas_part_data'):
    '''
    save the variables associated with the CanvasPart object 
    '''
    file_path = os.path.join(var.DATA_PATH, file_name + '.pickle')
    with open(file_path, 'wb') as handle:
        pickle.dump([canvas_comp,
                    time_inds_list_comp,
                    times,
                    file_size_bmp,
                    file_size_png],
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_canvas_part_time_steps(file_name='canvas_part_data'):
    '''
    load the variables associated with the CanvasPart object 
    '''
    file_path = os.path.join(var.DATA_PATH, file_name + '.pickle')
    with open(file_path, 'rb') as f:
        canvas_part_parameters = pickle.load(f)

    return canvas_part_parameters

def load_canvas_part(file_name):
    '''
    load the CanvasPart stored in the pickle object in this file_name 
    '''
    file_path = os.path.join(var.DATA_PATH, 'CanvasPart_' + file_name + '.pickle')
    with open(file_path, 'rb') as f:
        canpart = pickle.load(f)

    return canpart
