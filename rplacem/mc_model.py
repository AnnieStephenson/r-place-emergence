import cProfile
import os
import sys
import numpy as np
from rplacem import var as var
import pickle
from rplacem import utilities as util
import scipy


# GLOBAL
x_max_q1 = var.CANVAS_MINMAX[0][0][1]
x_max_q12 = var.CANVAS_MINMAX[1][0][1]
x_max_q1234 = var.CANVAS_MINMAX[2][0][1]
y_max_q1 = var.CANVAS_MINMAX[0][1][1]
y_max_q12 = var.CANVAS_MINMAX[1][1][1]
y_max_q1234 = var.CANVAS_MINMAX[2][1][1]
tot_pix = (x_max_q1234 + 1) * (y_max_q1234 + 1)

def get_all_pixel_changes_mc(data_file=var.FULL_DATA_FILE, 
                             random_pos=True,
                             random_col=True):
    '''
    TODO: add random_times option
    load all the pixel change data, modify it for mc null model testing, 
    and put it in a numpy array for easy access

    parameters
    ----------
    data_file : string that ends in .npz, optional

    returns
    -------
    pixel_changes_all : numpy structured array
            Contains all of the pixel change data from the entire dataset
    '''
    pixel_changes_all_npz = np.load(os.path.join(var.DATA_PATH, data_file))

    # save pixel changes as a structured array
    pixel_changes_all = np.zeros(len(pixel_changes_all_npz['seconds']),
                                 dtype=np.dtype([('seconds', np.float64),
                                                 ('xcoor', np.int16),
                                                 ('ycoor', np.int16),
                                                 ('user', np.uint32),
                                                 ('color', np.uint8),
                                                 ('moderator', np.bool_)])
                                 )

    pixel_changes_all['seconds'] = np.array(pixel_changes_all_npz['seconds'])
    pixel_changes_all['xcoor'] = np.array(pixel_changes_all_npz['pixelXpos'])
    pixel_changes_all['ycoor'] = np.array(pixel_changes_all_npz['pixelYpos'])

    if random_pos:
        x_min = np.min(pixel_changes_all['xcoor'])
        y_min = np.min(pixel_changes_all['ycoor'])
        x_max_q1 = var.CANVAS_MINMAX[0][0][1]
        x_max_q12 = var.CANVAS_MINMAX[1][0][1]
        x_max_q1234 = var.CANVAS_MINMAX[2][0][1]
        y_max_q1 = var.CANVAS_MINMAX[0][1][1]
        y_max_q12 = var.CANVAS_MINMAX[1][1][1]
        y_max_q1234 = var.CANVAS_MINMAX[2][1][1]

        # randomize in first quarter
        bool_inds_q1 = pixel_changes_all['seconds'] < var.TIME_ENLARGE[1]
        num_inds_q1 = np.sum(bool_inds_q1)
        pixel_changes_all['xcoor'][bool_inds_q1]= np.random.choice(np.arange(x_min, x_max_q1 + 1), size=num_inds_q1)
        pixel_changes_all['ycoor'][bool_inds_q1]= np.random.choice(np.arange(y_min, y_max_q1 + 1), size=num_inds_q1)

        # randomize in first and second quarter
        bool_inds_q12 = (pixel_changes_all['seconds'] >= var.TIME_ENLARGE[1]) & (pixel_changes_all['seconds'] < var.TIME_ENLARGE[2])
        num_inds_q12 = np.sum(bool_inds_q12)
        pixel_changes_all['xcoor'][bool_inds_q12] = np.random.choice(np.arange(x_min, x_max_q12 + 1), size = num_inds_q12)
        pixel_changes_all['ycoor'][bool_inds_q12] = np.random.choice(np.arange(y_min, y_max_q12 + 1), size = num_inds_q12)

        # randomize on entire canvas      
        bool_inds_q1234 = pixel_changes_all['seconds'] >= var.TIME_ENLARGE[2]
        num_inds_q1234 = np.sum(bool_inds_q1234)
        pixel_changes_all['xcoor'][bool_inds_q1234] = np.random.choice(np.arange(x_min, x_max_q1234 + 1), size=num_inds_q1234)
        pixel_changes_all['ycoor'][bool_inds_q1234] = np.random.choice(np.arange(y_min, y_max_q1234 + 1), size=num_inds_q1234)
    
    if random_col:
        # get the indices for each color choice range
        bool_inds_16col = (pixel_changes_all['seconds'] >= var.TIME_16COLORS) & (pixel_changes_all['seconds'] < var.TIME_24COLORS)
        bool_inds_24col = (pixel_changes_all['seconds'] >= var.TIME_24COLORS) & (pixel_changes_all['seconds'] < var.TIME_32COLORS)
        bool_inds_32col = (pixel_changes_all['seconds'] >= var.TIME_32COLORS) & (pixel_changes_all['seconds'] < var.TIME_WHITEOUT)

        # count the number of indices for each condition
        num_inds_16col = np.sum(bool_inds_16col)
        num_inds_24col = np.sum(bool_inds_24col)
        num_inds_32col = np.sum(bool_inds_32col)

        # randomize colors
        pixel_changes_all['color'][bool_inds_16col]= np.random.choice(np.arange(0, 16, dtype='uint8'), size=num_inds_16col)
        pixel_changes_all['color'][bool_inds_24col]= np.random.choice(np.arange(0, 24, dtype='uint8'), size=num_inds_24col)
        pixel_changes_all['color'][bool_inds_32col]= np.random.choice(np.arange(0, 32, dtype='uint8'), size=num_inds_32col)
        

    else:
        pixel_changes_all['color'] = np.array(pixel_changes_all_npz['colorIndex'])
    pixel_changes_all['user'] = np.array(pixel_changes_all_npz['userIndex'])
    pixel_changes_all['moderator'] = np.array(pixel_changes_all_npz['moderatorEvent'])

    return pixel_changes_all


def get_pixel_comp_time_map(filepath='canvas_comps_feb27_14221.pkl'):
    '''
    Defines a simplified map of the canvas through time, where each pixel assigned an integer
    based on the composition to which it belongs. In this simplified map, there are overlaps
    This map has 2 dimenstions for x and y, and a 3rd dimension for time. The time dimension comes from
    the unique times mentioned in the atlas for when borders change, and ends up being roughly every half hour
    '''
    # first check if the map is saved in the data folder
    map_name = 'canvas_time_map.pkl'
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..' ,'data')
    data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..' ,'data', map_name)
    files = os.listdir(data_path)
    if map_name in files:
        data_map = np.load(os.path.join(data_file_path), map_name, allow_pickle=True)
        with open(data_file_path , 'rb') as f:
            data_map = pickle.load(f)
        comp_pix_tm_map = data_map['comp_pix_tm_map']
        times_uniq = data_map['times_uniq']
        coords_comp_time_dict = data_map['coords_comp_time_dict']

    # if the map does not already exist, create it
    else:
        canvas_parts_file = os.path.join(os.getcwd(), filepath)

        with open(canvas_parts_file, "rb") as file:
            canvas_comp_list = pickle.load(file)

        times_uniq = np.array([])
        n_comps = len(canvas_comp_list)
        for i in range(n_comps):
            comp_coords_times = canvas_comp_list[i].coords_timerange
            comp_coords_times = np.concatenate(comp_coords_times).flatten()
            times_uniq = np.concatenate((times_uniq, np.unique(comp_coords_times)))
        times_uniq = np.unique(times_uniq)

        comp_pix_tm_map = -1*np.ones((2000, 2000, len(times_uniq)))
        veclen = np.vectorize(len)
        large_neg_num = -1000000
        n_comps = len(canvas_comp_list)
        n_pix_per_comp = np.zeros((n_comps, len(times_uniq))) # number of pixels in each composition at each time
        coords_comp_time_dict = {}
        for j in range(0, len(times_uniq)-1): 
            print('time index: ' + str(j))
            # Get the start and end time for this time step of the map
            tstart = times_uniq[j]
            tend = times_uniq[j + 1]
            for i in range(0, n_comps):
                # Get the array of coords for each composition: shape (n_coords)
                comp_coords = canvas_comp_list[i].coords 

                # Get the array of time_ranges for each composition, 
                # which contains a start and stop time for each coord, shape (n_coords, 2)
                comp_coords_times = canvas_comp_list[i].coords_timerange

                # Check is each coordinate has one time range (len(shape)==3) or multiple (len(shape)==n_coords))
                # Then for each case, find for which coordinates the time range of the coordinate overlaps with the current time step
                if len(comp_coords_times.shape)==3:
                    coord_inds_in_time = np.where((comp_coords_times[:,:,0].flatten()<=tstart) & (comp_coords_times[:,:,1].flatten()>=tend))[0]
                else:
                    max_numtr = np.max(veclen(comp_coords_times))
                    comp_coords_times_new = large_neg_num*np.ones((len(comp_coords_times), 2*max_numtr))

                    for k in range(len(comp_coords_times)):
                        comp_coords_times_1d = np.array(comp_coords_times[k]).flatten()
                        comp_coords_times_new[k, 0:len(comp_coords_times_1d)] = np.array(comp_coords_times[k]).flatten()

                    coord_inds_in_time = np.array([])
                    for m in range(0, comp_coords_times_new.shape[1], 2):
                        inds = np.where((comp_coords_times_new[:,m] <= tstart) & (comp_coords_times_new[:, m+1]>=tend))[0]
                        if len(inds)>0:
                            coord_inds_in_time = np.concatenate((coord_inds_in_time,inds))
                    coord_inds_in_time = coord_inds_in_time.astype(int) 

                # selects the x and y coords that are in the time range
                coords_to_set_to_i = comp_coords[:,coord_inds_in_time]

                # count the number of pixels in each composition
                n_pix_per_comp[i, j] = len(coord_inds_in_time)

                # put the coordinates for this composition at this time in a dictionary for easy access
                # keeps the coords as a 2d array
                coords_comp_time_dict[(i, j)] = coords_to_set_to_i

                # Set the pixel coordinate in the map to the composition number
                # This will overwrite if the pixel is already taken by a previous composition
                # It will use the last composition in the loop as the composition, which is arbitrary
                # TODO add another dimension of map to account for overlaps 
                comp_pix_tm_map[coords_to_set_to_i[0, :], coords_to_set_to_i[1, :], j] = i
            # find where the map is still -1
            coords_no_comp_x, coords_no_comp_y = np.where(comp_pix_tm_map[:, :, j] == -1)
            coords_comp_time_dict[(-1, j)] = np.vstack((coords_no_comp_x, coords_no_comp_y))

    to_save = {'comp_pix_tm_map': comp_pix_tm_map, 'times_uniq': times_uniq, 'coords_comp_time_dict': coords_comp_time_dict}
    with open(data_file_path, 'wb') as f:
        pickle.dump(to_save, f)

    return comp_pix_tm_map, times_uniq, coords_comp_time_dict

def sample_prob_pix_time(prob):
    '''
    For models that can be formulated with a matrix of probabilities over the canvas and time,
    this function samples that probabitlity matrix. 
    '''


def calc_coords_model(model_type = 'random', filepath_comps='canvas_comps_feb27_14221.pkl',
                      seed_real_changes=False):
    '''
    takes a model_type as input, and returns the pixel_changes_all array with modified coordinates for the models
    '''

    pixel_changes_all = util.get_all_pixel_changes() 

    if model_type == 'random':
        # uniform distribution of pixel changes
        # the boundaries of the uniform dist changes in time to accomodate the increasing canvas size
        # when not uniform, the probabiltiy is zero
        pixel_changes_model = sample_pixch_quarters(pixel_changes_all)

    if model_type == 'area':
        # find the area of each composition and sample coordinates proportional to size
        comp_pix_tm_map, times_uniq = get_pixel_comp_time_map(filepath=filepath_comps)
        pixel_changes_model = calc_coords_area(pixel_changes_all, comp_pix_tm_map, times_uniq)

    if model_type == 'loyalty': 
        # find all the users' first pixel changes
        # replace them with random pixel changes
        # all other pixel changes will be uniformly distributed #
        comp_pix_tm_map, times_uniq = get_pixel_comp_time_map(filepath=filepath_comps)
        pixel_changes_model = calc_coords_loyalty(pixel_changes_all, comp_pix_tm_map, times_uniq)

    if model_type == 'popularity': 
        # find all the users' first pixel changes
        comp_pix_tm_map, times_uniq = get_pixel_comp_time_map(filepath=filepath_comps)
        pixel_changes_model = calc_coords_popularity(pixel_changes_all, comp_pix_tm_map, times_uniq)

    return pixel_changes_model

def sample_pixch_quarters(pixel_changes, inds_to_sample=None, prob=None, coords_sample_from=None):
    '''
    Takes the pixel_changes_all array and returns boolean indices for each quarter of time
    '''
    

    if inds_to_sample is not None:
        pixel_changes_indexed = pixel_changes[inds_to_sample]
    else:
        pixel_changes_indexed = pixel_changes

    if coords_sample_from is None:
        x_coords_q1 = np.arange(0, x_max_q1 + 1)
        y_coords_q1 = np.arange(0, y_max_q1 + 1)
        x_coords_q12 = np.arange(0, x_max_q12 + 1)
        y_coords_q12 = np.arange(0, y_max_q12 + 1)
        x_coords_q1234 = np.arange(0, x_max_q1234 + 1)
        y_coords_q1234 = np.arange(0, y_max_q1234 + 1)
    else: 
        x_coords_q1 = coords_sample_from[0]
        y_coords_q1 = coords_sample_from[1]
        x_coords_q12 = coords_sample_from[2]
        y_coords_q12 = coords_sample_from[3]
        x_coords_q1234 = coords_sample_from[4]
        y_coords_q1234 = coords_sample_from[5]

    bool_inds_q1 = pixel_changes_indexed['seconds'] < var.TIME_ENLARGE[1]
    bool_inds_q12 = (pixel_changes_indexed['seconds'] >= var.TIME_ENLARGE[1]) & (pixel_changes_indexed['seconds'] < var.TIME_ENLARGE[2])
    bool_inds_q1234 = pixel_changes_indexed['seconds'] >= var.TIME_ENLARGE[2]


    pixel_changes_indexed['xcoor'][bool_inds_q1] = np.random.choice(x_coords_q1, size=np.sum(bool_inds_q1), p=prob)
    pixel_changes_indexed['ycoor'][bool_inds_q1] = np.random.choice(y_coords_q1, size=np.sum(bool_inds_q1), p=prob)

    pixel_changes_indexed['xcoor'][bool_inds_q12] = np.random.choice(x_coords_q12, size=np.sum(bool_inds_q12), p=prob)
    pixel_changes_indexed['ycoor'][bool_inds_q12] = np.random.choice(y_coords_q12, size=np.sum(bool_inds_q12), p=prob)

    pixel_changes_indexed['xcoor'][bool_inds_q1234] = np.random.choice(x_coords_q1234, size=np.sum(bool_inds_q1234), p=prob)
    pixel_changes_indexed['ycoor'][bool_inds_q1234] = np.random.choice(y_coords_q1234, size=np.sum(bool_inds_q1234), p=prob)

    pixel_changes[inds_to_sample] = pixel_changes_indexed
    return pixel_changes

def calc_coords_popularity(pixel_changes_all, times_uniq, seed_real_changes=False):
    '''
    calculates prob at each unique time
    and samples it
    '''
    prob = np.zeros((tot_pix, len(times_uniq)))
    num_arb = 20000.
    coords_flat = np.arange(0, tot_pix)
    pixel_changes_popularity = pixel_changes_all.copy()
    for i in range(len(times_uniq)-1):
        print('time: ' + str(i))
        tstart = times_uniq[i]
        tend = times_uniq[i + 1]
        t_inds = np.where((pixel_changes_popularity['seconds'] >= tstart) & (pixel_changes_popularity['seconds'] < tend))[0]

        if (times_uniq[i] == 0) or (times_uniq[i] == var.TIME_ENLARGE[1]) or (times_uniq[i] == var.TIME_ENLARGE[2]):
            print(times_uniq[i])
            # need to generate randomly for the first time step.
            if not seed_real_changes:
                pixel_changes_popularity = sample_pixch_quarters(pixel_changes_popularity, 
                                                                inds_to_sample=t_inds)
                                                                #coords_sample_from=[np.arange(0, x_max_q1 + 1),
                                                                #                    np.arange(0, y_max_q1 + 1),
                                                                #                    np.arange(x_max_q1, x_max_q12 + 1),
                                                                #                    np.arange(0, y_max_q12 + 1),
                                                                #                    np.arange(0, x_max_q1234 + 1),
                                                                #                    np.arange(y_max_q12, y_max_q1234 + 1)])


        else:
            # probability is proportional to the number of changes in the previous 30 min
            # so need to count the number of pixel changes that happened for each pixel in the previous 30 min
            coords_comb = pixel_changes_popularity['xcoor'] + num_arb * pixel_changes_popularity['ycoor']
            coords_unique, coord_counts = np.unique(coords_comb[t_inds_prev], return_counts=True)
            uni_x = coords_unique % num_arb
            uni_y = coords_unique // num_arb

            # put the counts in a 2d array of the canvas with zeros anywhere there are no counts 
            # TODO: need to make there be a small 0.1 probability for pixels with no changes, 
            # but it should still be zero if that part of canvas is not yet open. 
            # TODO: at the first 30 min of canvas section opening, the probability should be uniform across that new area of canvas
            #       but should be popularity driven accross the rest of the canvas? Not sure
            prob_sq = np.zeros((x_max_q1234 + 1, y_max_q1234 + 1))
            prob_sq[uni_x.astype(int), uni_y.astype(int)] = coord_counts
            prob[:,i] = prob_sq.flatten()
            prob[:,i] = prob[:,i]/np.sum(prob[:,i])
            
            # sample flattened coords
            samp_coord_inds = np.random.choice(coords_flat, size=len(t_inds), p=prob[:,i]) 
            
            # translate the coords back to x and y
            samp_coords_x = samp_coord_inds // (x_max_q1234 + 1) 
            samp_coords_y = samp_coord_inds % (y_max_q1234 + 1) 

            # set the new sampled x and y coords
            pixel_changes_popularity['xcoor'][t_inds] = samp_coords_x
            pixel_changes_popularity['ycoor'][t_inds] = samp_coords_y
        
        t_inds_prev = t_inds

    return pixel_changes_popularity


def calc_coords_loyalty(pixel_changes_all, comp_pix_tm_map, times_uniq, coords_comp_time_dict, seed_real_changes=False):
    '''
    Finds the first pixel change of each user, and replaces it with a random pixel change
    replace all the other pixel changes with random pixel changes within the same composition of their first choice. 
    '''
        
    pix_changes_user_sorted = np.sort(pixel_changes_all, order='user')
    inds_new_user = np.where(np.diff(pix_changes_user_sorted['user']) > 0)[0] + 1
    inds_new_user = np.concatenate(([0], inds_new_user))

    # sample the first pixel change of each user randomly
    if not seed_real_changes:
        pix_changes_user_sorted = sample_pixch_quarters(pix_changes_user_sorted, inds_to_sample=inds_new_user)

    # get the time indices of the first change of each user
    sorted_tm_indices = np.arange(0,len(times_uniq))
    first_change_time_int = (pix_changes_user_sorted['seconds'][inds_new_user] // 1800) * 1800
    indices_in_tm = np.searchsorted(times_uniq, first_change_time_int)
    t_ind = sorted_tm_indices[indices_in_tm]

    # get the composition number where each user made their first change
    comp_user = comp_pix_tm_map[pix_changes_user_sorted['xcoor'][inds_new_user], pix_changes_user_sorted['ycoor'][inds_new_user], t_ind]

    for i in range(0,len(inds_new_user)):
        if i % 10000 ==0:
            print('user number: ' + str(i)) 
        coords_comp_t = coords_comp_time_dict[(comp_user[i], t_ind[i])]
        len_coords = coords_comp_t.shape[1]
        if i < len(inds_new_user)-1:
            end_change_ind = inds_new_user[i+1]
        else:
            end_change_ind = len(pixel_changes_all)
        num_changes = end_change_ind - inds_new_user[i]
        coords_inds_sampled = np.random.choice(np.arange(0,len_coords), size = num_changes)
        pix_changes_user_sorted['xcoor'][inds_new_user[i]: end_change_ind] = coords_comp_t[0, coords_inds_sampled]
        pix_changes_user_sorted['ycoor'][inds_new_user[i]: end_change_ind] = coords_comp_t[1, coords_inds_sampled]

    # re-sort according to time
    pixel_changes_loyalty = np.sort(pix_changes_user_sorted, order='seconds')

    return pixel_changes_loyalty


def calc_coords_area(pixel_changes_all, comp_pix_tm_map, times_uniq):
    '''
    Gets the pixel changes assuming that each pixel probability is weighted
    by the area of the composition that contains it
    '''
    prob = np.zeros((tot_pix, len(times_uniq)))
    coords_flat = np.arange(0, tot_pix)
    pixel_changes_area = pixel_changes_all.copy()
    for i in range(len(times_uniq)-1):
        print(i)
        counts = np.bincount(comp_pix_tm_map[:, :, i].flatten().astype('int') + 1)
        prob[:,i] = counts[comp_pix_tm_map[:, :, i].flatten().astype('int') + 1]
        prob[prob[:,i] == counts[0]] = 0
        tstart = times_uniq[i]
        tend = times_uniq[i + 1]
        t_inds = np.where((pixel_changes_all['seconds'] >= tstart) & (pixel_changes_all['seconds'] < tend))[0]
        
        # sample flattened coords
        samp_coord_inds = np.random.choice(coords_flat, size = len(t_inds), p = prob[:,i]/np.sum(prob[:,i]))
        
        # translate the coords back to x and y
        samp_coords_x = samp_coord_inds // (x_max_q1234 + 1) 
        samp_coords_y = samp_coord_inds % (y_max_q1234 + 1) 

        # set the new sampled x and y coords
        pixel_changes_area['xcoor'][t_inds] = samp_coords_x
        pixel_changes_area['ycoor'][t_inds] = samp_coords_y
    return pixel_changes_area

def get_all_pixel_changes_mc():
    '''
    Calls the previous 3 functions
    Takes a model_type as input to pass to calc_prob_pix_time()
    Returns the pixel_changes_all array
    '''