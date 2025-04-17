import cProfile
import os
import sys
import numpy as np
from rplacem import var as var
import pickle
from rplacem import utilities as util
import scipy
import matplotlib.pyplot as plt


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


def get_pixel_comp_time_map(filepath='canvas_comps_feb27_14221.pkl',
                            times_before_whiteout=True,
                            num_layers=2):
    '''
    Defines a simplified map of the canvas through time, where each pixel assigned an integer
    based on the composition to which it belongs. In this simplified map, there are overlaps
    This map has 2 dimenstions for x and y, and a 3rd dimension for time. The time dimension comes from
    the unique times mentioned in the atlas for when borders change, and ends up being roughly every half hour
    '''
    # first check if the map is saved in the data folder
    map_name = 'canvas_time_map_' + str(num_layers) + '_layers.pkl'
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                             'data')
    data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'data', map_name)
    files = os.listdir(data_path)
    if map_name in files:
        #data_map = np.load(os.path.join(data_file_path), map_name, allow_pickle=True)
        with open(data_file_path, 'rb') as f:
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
        canvas_comp_list.sort(
            key=lambda x: x.num_pix(),
            reverse=True)  # sort by size of canvas part, largest first
        n_comps = len(canvas_comp_list)
        for i in range(n_comps):
            comp_coords_times = canvas_comp_list[i].coords_timerange
            comp_coords_times = np.concatenate(comp_coords_times).flatten()
            times_uniq = np.concatenate(
                (times_uniq, np.unique(comp_coords_times)))
        times_uniq = np.unique(times_uniq)
        if times_before_whiteout:
            times_uniq = np.concatenate(
                (times_uniq[times_uniq < var.TIME_WHITEOUT],
                 [var.TIME_WHITEOUT]))

        comp_pix_tm_map = -1 * np.ones(
            (2000, 2000, len(times_uniq), num_layers), dtype=np.int16)
        veclen = np.vectorize(len)
        large_neg_num = -1000000
        n_comps = len(canvas_comp_list)
        n_pix_per_comp = np.zeros(
            (n_comps, len(times_uniq)
             ))  # number of pixels in each composition at each time
        coords_comp_time_dict = {}
        for j in range(0, len(times_uniq) - 1):
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
                if len(comp_coords_times.shape) == 3:
                    coord_inds_in_time = np.where(
                        (comp_coords_times[:, :, 0].flatten() <= tstart)
                        & (comp_coords_times[:, :, 1].flatten() >= tend))[0]
                else:
                    max_numtr = np.max(veclen(comp_coords_times))
                    comp_coords_times_new = large_neg_num * np.ones(
                        (len(comp_coords_times), 2 * max_numtr))

                    for k in range(len(comp_coords_times)):
                        comp_coords_times_1d = np.array(
                            comp_coords_times[k]).flatten()
                        comp_coords_times_new[
                            k, 0:len(comp_coords_times_1d)] = np.array(
                                comp_coords_times[k]).flatten()

                    coord_inds_in_time = np.array([])
                    for m in range(0, comp_coords_times_new.shape[1], 2):
                        inds = np.where(
                            (comp_coords_times_new[:, m] <= tstart)
                            & (comp_coords_times_new[:, m + 1] >= tend))[0]
                        if len(inds) > 0:
                            coord_inds_in_time = np.concatenate(
                                (coord_inds_in_time, inds))
                    coord_inds_in_time = coord_inds_in_time.astype(int)

                # selects the x and y coords that are in the time range
                coords_to_set_to_i = comp_coords[:, coord_inds_in_time]

                # count the number of pixels in each composition
                n_pix_per_comp[i, j] = len(coord_inds_in_time)

                # put the coordinates for this composition at this time in a dictionary for easy access
                # keeps the coords as a 2d array, i is comp number and j is time index
                coords_comp_time_dict[(i, j)] = coords_to_set_to_i

                # Set the pixel coordinate in the map to the composition number
                # This will overwrite if the pixel is already taken by a previous composition
                # It will use the last composition in the loop as the composition, which is arbitrary

                # TODO
                # add another dimension of map to account for overlaps
                # for now, just account for overlap of 2 compositions; this should capture most cases.
                # if any values indexed using the line below are not -1, then there is an overlap (use np.isin?)
                # so fill the second dimension of the map with the second composition number.
                # if there is a third overlap, then the second dimension will be overwritten.

                # If all the coords for a composition are -1, no comp has been assigned to this layer.
                if num_layers == 2:
                    if np.all(comp_pix_tm_map[coords_to_set_to_i[0, :],
                                              coords_to_set_to_i[1, :], j,
                                              0] == -1):
                        comp_pix_tm_map[coords_to_set_to_i[0, :],
                                        coords_to_set_to_i[1, :], j, 0] = i
                    # If there is a composition already assigned in the first layer, set the second layer to the current composition
                    else:
                        comp_pix_tm_map[coords_to_set_to_i[0, :],
                                        coords_to_set_to_i[1, :], j, 1] = i

                elif num_layers == 1:
                    # You'll always just overwrite the first layer in this case
                    comp_pix_tm_map[coords_to_set_to_i[0, :],
                                    coords_to_set_to_i[1, :], j, 0] = i

            # find where the map is still -1
            coords_no_comp_x, coords_no_comp_y = np.where(
                comp_pix_tm_map[:, :, j, 0] == -1)
            coords_comp_time_dict[(-1, j)] = np.vstack(
                (coords_no_comp_x, coords_no_comp_y))

            if num_layers == 2:
                coords_no_comp_x, coords_no_comp_y = np.where(
                    comp_pix_tm_map[:, :, j, 1] == -1)
                # note that we use -2 to designate where there is a minus one on the map in the overlap dimension
                # not the most elegant solution, but it works for now
                coords_comp_time_dict[(-2, j)] = np.vstack(
                    (coords_no_comp_x, coords_no_comp_y))

    to_save = {
        'comp_pix_tm_map': comp_pix_tm_map,
        'times_uniq': times_uniq,
        'coords_comp_time_dict': coords_comp_time_dict
    }
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

def calc_coords_loyalty_comp(pixel_changes_all,
                        comp_pix_tm_map,
                        times_uniq,
                        coords_comp_time_dict,
                        seed_real_changes=False):
    """
    Reassigns pixel coordinates to simulate user loyalty behavior:
    
    - The first change of each user is either kept or randomly sampled (depending on `seed_real_changes`).
    - All subsequent changes by a user are relocated to random pixels within the same composition
      that they initially contributed to.

    Parameters:
        pixel_changes_all (structured np.ndarray): Array of all pixel changes, with fields 
            including 'user', 'seconds', 'xcoor', and 'ycoor'.
        comp_pix_tm_map (ndarray): 4D array mapping (x, y, time, layer) to composition IDs.
        times_uniq (ndarray): 1D array of unique time boundaries (in seconds).
        coords_comp_time_dict (dict): Maps (composition_id, time_index) to a tuple of (x_coords, y_coords).
        seed_real_changes (bool): If True, preserve the user's actual first change.
                                   If False, sample a new first change randomly.

    Returns:
        pixel_changes_loyalty (structured np.ndarray): Modified array of pixel changes with updated coordinates.
    """
    num_layers = comp_pix_tm_map.shape[-1]

    # Sort changes by user to easily find first change per user
    pix_changes_user_sorted = np.sort(pixel_changes_all, order='user')

    # Get indices where a new user starts
    inds_new_user = np.where(
        np.diff(pix_changes_user_sorted['user']) > 0)[0] + 1
    inds_new_user = np.concatenate(([0], inds_new_user))

    # Optionally replace users' first change with a random one
    if not seed_real_changes:
        pix_changes_user_sorted = sample_pixch_quarters(
            pix_changes_user_sorted, inds_to_sample=inds_new_user)

    # Get time bin index for each user's first change
    first_change_seconds = pix_changes_user_sorted['seconds'][inds_new_user]
    t_ind = np.searchsorted(times_uniq, first_change_seconds, side='right') - 1

    # Get composition index for each user's first change
    comp_user = comp_pix_tm_map[
        pix_changes_user_sorted['xcoor'][inds_new_user],
        pix_changes_user_sorted['ycoor'][inds_new_user], t_ind, 0]

    if num_layers > 1:
        missing_comps = (comp_user == -1)
        if np.any(missing_comps):
            # Try to search for comp in bottom layer
            bottom_layer_comps = comp_pix_tm_map[pix_changes_user_sorted['xcoor'][inds_new_user][missing_comps],
                                                pix_changes_user_sorted['ycoor'][inds_new_user][missing_comps],
                                                t_ind[missing_comps], 1]

            # Replace comp_user values where comp_user == -1
            comp_user[missing_comps] = bottom_layer_comps

            # Set any -2s back to -1
            comp_user[comp_user == -2] = -1

    # Reassign coordinates for all of each user's changes to random pixels within the same composition
    for i, start_change_ind in enumerate(inds_new_user):
        if i % 10000 == 0:
            print('User Number: ' + str(i) + ' out of ' + str(len(inds_new_user)))

        end_change_ind = inds_new_user[i + 1] if i < (len(inds_new_user) - 1) else len(pix_changes_user_sorted)
        user_change_inds = np.arange(start_change_ind, end_change_ind)

        loyalty_comp = None
        loyalty_time = None

        for j in range(user_change_inds.shape[0]):
            idx = user_change_inds[j]
            xj = pix_changes_user_sorted['xcoor'][idx]
            yj = pix_changes_user_sorted['ycoor'][idx]
            tj = np.searchsorted(times_uniq, [pix_changes_user_sorted['seconds'][idx]], side='right')[0] - 1

            compj = comp_pix_tm_map[xj, yj, tj, 0]
            if compj == -1 and num_layers > 1:
                compj = comp_pix_tm_map[xj, yj, tj, 1]

            if compj != -1:
                # Set loyalty comp and sample for all remaining changes
                loyalty_comp = compj
                loyalty_time = tj

                coords_comp_t = coords_comp_time_dict[(loyalty_comp, loyalty_time)]
                len_coords = coords_comp_t.shape[1]

                # Sample for remaining changes including this one
                num_remaining_changes = end_change_ind - idx
                coords_inds_sampled = np.random.choice(len_coords, size=num_remaining_changes)

                pix_changes_user_sorted['xcoor'][idx:end_change_ind] = coords_comp_t[0, coords_inds_sampled]
                pix_changes_user_sorted['ycoor'][idx:end_change_ind] = coords_comp_t[1, coords_inds_sampled]
                break  # Done with this user

    # re-sort according to time
    pixel_changes_loyalty = np.sort(pix_changes_user_sorted, order='seconds')

    return pixel_changes_loyalty


def calc_coords_loyalty(pixel_changes_all,
                        comp_pix_tm_map,
                        times_uniq,
                        coords_comp_time_dict,
                        seed_real_changes=False):
    """
    Reassigns pixel coordinates to simulate user loyalty behavior:
    
    - The first change of each user is either kept or randomly sampled (depending on `seed_real_changes`).
    - All subsequent changes by a user are relocated to random pixels within the same composition
      that they initially contributed to.

    Parameters:
        pixel_changes_all (structured np.ndarray): Array of all pixel changes, with fields 
            including 'user', 'seconds', 'xcoor', and 'ycoor'.
        comp_pix_tm_map (ndarray): 4D array mapping (x, y, time, layer) to composition IDs.
        times_uniq (ndarray): 1D array of unique time boundaries (in seconds).
        coords_comp_time_dict (dict): Maps (composition_id, time_index) to a tuple of (x_coords, y_coords).
        seed_real_changes (bool): If True, preserve the user's actual first change.
                                   If False, sample a new first change randomly.

    Returns:
        pixel_changes_loyalty (structured np.ndarray): Modified array of pixel changes with updated coordinates.
    """
    num_layers = comp_pix_tm_map.shape[-1]

    # Sort changes by user to easily find first change per user
    pix_changes_user_sorted = np.sort(pixel_changes_all, order='user')

    # Get indices where a new user starts
    inds_new_user = np.where(
        np.diff(pix_changes_user_sorted['user']) > 0)[0] + 1
    inds_new_user = np.concatenate(([0], inds_new_user))

    # Optionally replace users' first change with a random one
    if not seed_real_changes:
        pix_changes_user_sorted = sample_pixch_quarters(
            pix_changes_user_sorted, inds_to_sample=inds_new_user)

    # Get time bin index for each user's first change
    first_change_seconds = pix_changes_user_sorted['seconds'][inds_new_user]
    t_ind = np.searchsorted(times_uniq, first_change_seconds, side='right') - 1

    # Get composition index for each user's first change
    comp_user = comp_pix_tm_map[
        pix_changes_user_sorted['xcoor'][inds_new_user],
        pix_changes_user_sorted['ycoor'][inds_new_user], t_ind, 0]

    if num_layers > 1:
        missing_comps = (comp_user == -1)
        if np.any(missing_comps):
            # Try to search for comp in bottom layer
            bottom_layer_comps = comp_pix_tm_map[pix_changes_user_sorted['xcoor'][inds_new_user][missing_comps],
                                                pix_changes_user_sorted['ycoor'][inds_new_user][missing_comps],
                                                t_ind[missing_comps], 1]

            # Replace comp_user values where comp_user == -1
            comp_user[missing_comps] = bottom_layer_comps

            # Set any -2s back to -1
            comp_user[comp_user == -2] = -1

    # Reassign coordinates for all of each user's changes to random pixels within the same composition
    for i, start_change_ind in enumerate(inds_new_user):
        if i % 10000 == 0:
            print('User Number: ' + str(i) + ' out of ' + str(len(inds_new_user)))

        coords_comp_t = coords_comp_time_dict[(comp_user[i], t_ind[i])]
        len_coords = coords_comp_t.shape[1]

        # Determine index of last change by this user
        end_change_ind = inds_new_user[i + 1] if i < (len(inds_new_user) - 1) else len(pix_changes_user_sorted)
        num_changes = end_change_ind - start_change_ind

        coords_inds_sampled = np.random.choice(len_coords, size=num_changes) # randomly sample within the composition
        pix_changes_user_sorted['xcoor'][start_change_ind:end_change_ind] = coords_comp_t[0, coords_inds_sampled]
        pix_changes_user_sorted['ycoor'][start_change_ind:end_change_ind] = coords_comp_t[1, coords_inds_sampled]

    # re-sort according to time
    pixel_changes_loyalty = np.sort(pix_changes_user_sorted, order='seconds')

    return pixel_changes_loyalty

def calc_coords_loyalty_old(pixel_changes_all, comp_pix_tm_map, times_uniq, coords_comp_time_dict, seed_real_changes=False):
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
    comp_user = comp_pix_tm_map[pix_changes_user_sorted['xcoor'][inds_new_user], pix_changes_user_sorted['ycoor'][inds_new_user], t_ind, 0]

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



def calc_coords_area(pixel_changes_all,
                     comp_pix_tm_map,
                     coords_comp_time_dict,
                     times_uniq,
                     exp=1, const=1):
    """
    Reassigns pixel coordinates in `pixel_changes_all` based on a probability 
    distribution weighted by the area (to a power) of compositions containing each pixel.

    Parameters:
        pixel_changes_all (structured np.array): Array containing pixel change data,
            including 'seconds', 'xcoor', and 'ycoor' fields.
        comp_pix_tm_map (ndarray): 4D array of shape (H, W, T, K) mapping pixels to
            composition IDs over time (T) and overlap index (K).
        coords_comp_time_dict (dict): Dictionary mapping (composition_index, time_index)
            to (x_coords, y_coords) of composition pixels.
        times_uniq (ndarray): Sorted 1D array of unique time values.
        exp (float, optional): Exponent applied to the composition area weights. 
            Default is 1 (no weighting effect).

    Returns:
        pixel_changes_area (structured np.array): Modified copy of `pixel_changes_all`
            with updated x and y coordinates.
    """
    num_layers = comp_pix_tm_map.shape[-1]
    coords_flat = np.arange(tot_pix)
    pixel_changes_area = pixel_changes_all.copy()
    prob = np.zeros((tot_pix, num_layers))

    for i in range(len(times_uniq) - 1):
        print(f"Processing time interval {i + 1}/{len(times_uniq) - 1}")
        for k in range(num_layers):
            map_t_ov = comp_pix_tm_map[:, :, i, k]
            flat_ids = map_t_ov.ravel()

            # Add 1 to avoid bincount issues with -1 and -2 (assumes background = -1 or -2)
            counts = np.bincount(flat_ids + 1)
            prob[:, k] = counts[flat_ids + 1]
            coords_x, coords_y = coords_comp_time_dict[(-1 - k, i)]
            coords_comb_nocomp = coords_y.astype(
                'int') + coords_x.astype('int') * (x_max_q1234 + 1)
            prob[coords_comb_nocomp, k] = 0
            prob[:, k] = prob[:, k]**exp + const # add the + const so that the no comps have some probability

        # Select the pixel changes in the time range
        tstart = times_uniq[i]
        tend = times_uniq[i + 1]
        t_inds = np.where((pixel_changes_all['seconds'] >= tstart)
                          & (pixel_changes_all['seconds'] < tend))[0]

        # sample flattened coords
        prob_sum_ov = np.sum(prob, axis=1)
        samp_coord_inds = np.random.choice(coords_flat,
                                           size=len(t_inds),
                                           p=prob_sum_ov / prob_sum_ov.sum())

        # translate the coords back to x and y
        samp_coords_x = samp_coord_inds // (x_max_q1234 + 1)
        samp_coords_y = samp_coord_inds % (y_max_q1234 + 1)

        # set the new sampled x and y coords
        pixel_changes_area['xcoor'][t_inds] = samp_coords_x
        pixel_changes_area['ycoor'][t_inds] = samp_coords_y
    return pixel_changes_area


def calc_coords_perimeter(pixel_changes_all, comp_pix_tm_map, times_uniq,
                          coords_comp_time_dict, const=1):
    """
    Reassigns pixel coordinates in `pixel_changes_all` by sampling from the perimeter 
    of compositions over time.

    For each time bin, the probability of sampling a pixel is proportional 
    to the size of the perimeter of the composition it belongs to. The perimeter is 
    computed using binary erosion.

    Parameters:
        pixel_changes_all (structured np.ndarray): Original pixel change data with fields 
            including 'seconds', 'xcoor', and 'ycoor'.
        comp_pix_tm_map (np.ndarray): 4D array of shape (H, W, T, K), mapping each pixel to a 
            composition index over time T and overlap dimension K.
        times_uniq (np.ndarray): 1D array of time boundaries (in seconds).
        coords_comp_time_dict (dict): Maps (composition_id, time_index) to a tuple of 
            (x_coords, y_coords) of all pixels in that composition.

    Returns:
        pixel_changes_perim (structured np.ndarray): Updated copy of `pixel_changes_all` 
            with resampled x and y coordinates.
    """
    num_layers = comp_pix_tm_map.shape[-1]
    prob = np.zeros((tot_pix, num_layers))
    coords_flat = np.arange(0, tot_pix)
    pixel_changes_perim = pixel_changes_all.copy()

    for t in range(len(times_uniq) - 1):
        print(f"Processing time index {t}/{len(times_uniq) - 2}")
        i = 0
        for k in range(num_layers):  # overlap dimension
            if t > 0:
                diff = comp_pix_tm_map[:, :, t - 1, k] - comp_pix_tm_map[:, :, t, k]
                indsx, indsy = np.where(diff != 0)
                c_ints = np.unique(comp_pix_tm_map[indsx, indsy, t, k])
            else:
                c_ints = np.unique(comp_pix_tm_map[:, :, t, k])
            for c in c_ints:
                if c == -1:
                    coords_x, coords_y = coords_comp_time_dict[(c - k, t)] # c-k means we index -1 in 1st layer and -2 when in 2nd layer
                    coords_comb = coords_y.astype(
                        'int64') + coords_x.astype('int64') * (x_max_q1234 + 1)
                    prob[coords_comb, k] = const
                else:
                    coords_x, coords_y = coords_comp_time_dict[(c, t)]
                    coords_comb = coords_y.astype('int64') + coords_x.astype('int64') * (x_max_q1234 + 1)
                    mask = (comp_pix_tm_map[:, :, t, k] == c)
                    eroded_mask = scipy.ndimage.binary_erosion(mask)
                    perimeter_mask = mask & ~eroded_mask
                    perimeter_count = np.sum(perimeter_mask)
                    prob[coords_comb, k] = perimeter_count + const

                i += 1

        tstart = times_uniq[t]
        tend = times_uniq[t + 1]
        t_inds = np.where((pixel_changes_all['seconds'] >= tstart)
                          & (pixel_changes_all['seconds'] < tend))[0]
        prob_sum_ov = np.sum(prob, axis=1)
        samp_coord_inds = np.random.choice(coords_flat,
                                           size=len(t_inds),
                                           p=prob_sum_ov / np.sum(prob_sum_ov))

        # translate the coords back to x and y
        samp_coords_x = samp_coord_inds // (x_max_q1234 + 1)
        samp_coords_y = samp_coord_inds % (y_max_q1234 + 1)

        # set the new sampled x and y coords
        pixel_changes_perim['xcoor'][t_inds] = samp_coords_x
        pixel_changes_perim['ycoor'][t_inds] = samp_coords_y

    return pixel_changes_perim

    #def get_all_pixel_changes_mc():
    '''
    Calls the previous 3 functions
    Takes a model_type as input to pass to calc_prob_pix_time()
    Returns the pixel_changes_all array
    '''
