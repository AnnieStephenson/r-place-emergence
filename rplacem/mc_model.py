import os
import numpy as np
from rplacem import var as var
import pickle
from scipy.spatial import ConvexHull, QhullError
from rplacem import utilities as util


# GLOBALS
# Compute canvas bounds across all expansion states
x_min = int(var.CANVAS_MINMAX[:, 0, 0].min())
y_min = int(var.CANVAS_MINMAX[:, 1, 0].min())
x_max = int(var.CANVAS_MINMAX[:, 0, 1].max())
y_max = int(var.CANVAS_MINMAX[:, 1, 1].max())
x_dim = x_max - x_min + 1
y_dim = y_max - y_min + 1
tot_pix = x_dim * y_dim
_coords_flat = np.arange(tot_pix)


def sample_prob_pix_time(prob, pixel_changes, t_inds):
    '''
    Sample pixel coordinates from a probability distribution and assign them.
    All coordinates are 0-based (caller handles real<->0-based offset).

    Parameters
    ----------
    prob : ndarray, shape (tot_pix,)
        Probability weights over the flattened canvas (need not be normalized).
        Flat index convention: flat = x_0based * y_dim + y_0based.
    pixel_changes : structured ndarray
        Array with 'xcoor' and 'ycoor' fields, modified in-place.
    t_inds : ndarray
        Indices into pixel_changes to resample.
    '''
    prob_norm = prob / prob.sum()
    samp_inds = np.random.choice(_coords_flat, size=len(t_inds), p=prob_norm)
    pixel_changes['xcoor'][t_inds] = samp_inds // y_dim
    pixel_changes['ycoor'][t_inds] = samp_inds % y_dim


def sample_pixch_quarters(pixel_changes, inds_to_sample=None, prob=None, coords_sample_from=None):
    '''
    Randomize pixel coordinates respecting canvas size changes over time.
    Operates in real coordinate space.

    Parameters
    ----------
    pixel_changes : structured ndarray
        Array with 'seconds', 'xcoor', 'ycoor' fields.
    inds_to_sample : ndarray, optional
        If given, only resample these indices.
    prob : ndarray, optional
        Probability weights for sampling (must match coordinate array length).
    coords_sample_from : list, optional
        Alternating x_coords, y_coords arrays for each canvas state.
    '''
    if inds_to_sample is not None:
        pixel_changes_indexed = pixel_changes[inds_to_sample]
    else:
        pixel_changes_indexed = pixel_changes

    x_coords_list = []
    y_coords_list = []

    if coords_sample_from is None:
        for canvas in var.CANVAS_MINMAX:
            x_coords_list.append(np.arange(canvas[0][0], canvas[0][1] + 1))
            y_coords_list.append(np.arange(canvas[1][0], canvas[1][1] + 1))
    else:
        x_coords_list = coords_sample_from[::2]
        y_coords_list = coords_sample_from[1::2]

    time_enlarges = var.TIME_ENLARGE
    for i in range(len(time_enlarges)):
        if i + 1 < len(time_enlarges):
            bool_inds = (pixel_changes_indexed['seconds'] >= time_enlarges[i]) & (pixel_changes_indexed['seconds'] < time_enlarges[i + 1])
        else:
            bool_inds = pixel_changes_indexed['seconds'] >= time_enlarges[i]

        num_inds = np.sum(bool_inds)
        pixel_changes_indexed['xcoor'][bool_inds] = np.random.choice(x_coords_list[i], size=num_inds, p=prob)
        pixel_changes_indexed['ycoor'][bool_inds] = np.random.choice(y_coords_list[i], size=num_inds, p=prob)

    if inds_to_sample is not None:
        pixel_changes[inds_to_sample] = pixel_changes_indexed
    else:
        pixel_changes = pixel_changes_indexed

    return pixel_changes


def get_pixel_comp_time_map(filepath='canvas_comps_feb27_14221.pkl',
                            times_before_whiteout=True,
                            num_layers=1):
    '''
    Build or load a map of the canvas through time, where each pixel is assigned
    an integer based on the composition it belongs to. Supports overlapping
    compositions via the num_layers parameter.

    All coordinates in the returned map and dict are 0-based
    (real coordinate minus x_min/y_min).

    Returns
    -------
    comp_pix_tm_map : ndarray, shape (x_dim, y_dim, n_times, num_layers)
    times_uniq : ndarray
    coords_comp_time_dict : dict mapping (comp_id, time_index) to (2, n_coords) 0-based array
    '''
    map_name = f'canvas_time_map_{num_layers}_layers_{var.year}.pkl'
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    data_file_path = os.path.join(data_path, map_name)

    if os.path.exists(data_file_path):
        with open(data_file_path, 'rb') as f:
            data_map = pickle.load(f)
        comp_pix_tm_map = data_map['comp_pix_tm_map']
        times_uniq = data_map['times_uniq']
        coords_comp_time_dict = data_map['coords_comp_time_dict']

    else:
        canvas_parts_file = os.path.join(os.getcwd(), filepath)
        with open(canvas_parts_file, "rb") as file:
            canvas_comp_list = pickle.load(file)

        canvas_comp_list.sort(key=lambda x: x.num_pix(), reverse=True)
        n_comps = len(canvas_comp_list)
        all_times = []
        for i in range(n_comps):
            comp_coords_times = canvas_comp_list[i].coords_timerange
            comp_coords_times = np.concatenate(comp_coords_times).flatten()
            all_times.append(np.unique(comp_coords_times))
        times_uniq = np.unique(np.concatenate(all_times))
        if times_before_whiteout:
            times_uniq = np.concatenate(
                (times_uniq[times_uniq < var.TIME_WHITEOUT],
                 [var.TIME_WHITEOUT]))

        comp_pix_tm_map = -1 * np.ones(
            (x_dim, y_dim, len(times_uniq), num_layers), dtype=np.int16)
        veclen = np.vectorize(len)
        large_neg_num = -1000000
        n_comps = len(canvas_comp_list)
        n_pix_per_comp = np.zeros((n_comps, len(times_uniq)))
        coords_comp_time_dict = {}

        for j in range(0, len(times_uniq) - 1):
            print('time index: ' + str(j))
            tstart = times_uniq[j]
            tend = times_uniq[j + 1]

            for i in range(0, n_comps):
                comp_coords = canvas_comp_list[i].coords
                comp_coords_times = canvas_comp_list[i].coords_timerange

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
                            k, 0:len(comp_coords_times_1d)] = comp_coords_times_1d

                    coord_inds_list = []
                    for m in range(0, comp_coords_times_new.shape[1], 2):
                        inds = np.where(
                            (comp_coords_times_new[:, m] <= tstart)
                            & (comp_coords_times_new[:, m + 1] >= tend))[0]
                        if len(inds) > 0:
                            coord_inds_list.append(inds)
                    coord_inds_in_time = (np.concatenate(coord_inds_list).astype(int)
                                          if coord_inds_list
                                          else np.array([], dtype=int))

                # Get real coords and convert to 0-based for map indexing and dict storage
                coords_real = comp_coords[:, coord_inds_in_time]
                coords_0b = np.empty_like(coords_real)
                coords_0b[0] = coords_real[0] - x_min
                coords_0b[1] = coords_real[1] - y_min

                n_pix_per_comp[i, j] = len(coord_inds_in_time)
                coords_comp_time_dict[(i, j)] = coords_0b

                if num_layers == 2:
                    if np.all(comp_pix_tm_map[coords_0b[0],
                                              coords_0b[1], j,
                                              0] == -1):
                        comp_pix_tm_map[coords_0b[0],
                                        coords_0b[1], j, 0] = i
                    else:
                        comp_pix_tm_map[coords_0b[0],
                                        coords_0b[1], j, 1] = i

                elif num_layers == 1:
                    comp_pix_tm_map[coords_0b[0],
                                    coords_0b[1], j, 0] = i

            # Find pixels with no composition assigned (np.where returns 0-based indices)
            coords_no_comp_x, coords_no_comp_y = np.where(
                comp_pix_tm_map[:, :, j, 0] == -1)
            coords_comp_time_dict[(-1, j)] = np.vstack(
                (coords_no_comp_x, coords_no_comp_y))

            if num_layers == 2:
                coords_no_comp_x, coords_no_comp_y = np.where(
                    comp_pix_tm_map[:, :, j, 1] == -1)
                # -2 key designates no-comp in the overlap (second) layer
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


def calc_coords_random(pixel_changes_all, random_pos=True, random_col=True):
    '''
    Generate a fully random null model. Randomizes positions uniformly across
    the canvas (respecting size changes over time) and/or colors (respecting
    palette changes over time).

    Parameters
    ----------
    pixel_changes_all : structured ndarray
        Loaded via util.get_all_pixel_changes().
    random_pos : bool
        If True, randomize pixel positions.
    random_col : bool
        If True, randomize colors.
    '''
    pixel_changes = pixel_changes_all.copy()

    if random_pos:
        pixel_changes = sample_pixch_quarters(pixel_changes)

    if random_col:
        bool_inds_16col = ((pixel_changes['seconds'] >= var.TIME_16COLORS)
                           & (pixel_changes['seconds'] < var.TIME_24COLORS))
        bool_inds_24col = ((pixel_changes['seconds'] >= var.TIME_24COLORS)
                           & (pixel_changes['seconds'] < var.TIME_32COLORS))
        bool_inds_32col = ((pixel_changes['seconds'] >= var.TIME_32COLORS)
                           & (pixel_changes['seconds'] < var.TIME_WHITEOUT))

        pixel_changes['color'][bool_inds_16col] = np.random.choice(
            np.arange(0, 16, dtype='uint8'), size=np.sum(bool_inds_16col))
        pixel_changes['color'][bool_inds_24col] = np.random.choice(
            np.arange(0, 24, dtype='uint8'), size=np.sum(bool_inds_24col))
        pixel_changes['color'][bool_inds_32col] = np.random.choice(
            np.arange(0, 32, dtype='uint8'), size=np.sum(bool_inds_32col))

    return pixel_changes


def calc_coords_area(pixel_changes_all,
                     comp_pix_tm_map,
                     coords_comp_time_dict,
                     times_uniq,
                     exp=1):
    """
    Reassigns pixel coordinates based on a probability distribution weighted by
    the area (raised to exp) of compositions containing each pixel.
    Non-composition pixels are treated as single-pixel compositions (weight = 1).

    Parameters
    ----------
    pixel_changes_all : structured ndarray
    comp_pix_tm_map : ndarray, shape (x_dim, y_dim, n_times, num_layers)
    coords_comp_time_dict : dict
    times_uniq : ndarray
    exp : float
        Exponent applied to composition area weights.
    """
    num_layers = comp_pix_tm_map.shape[-1]
    pixel_changes_area = pixel_changes_all.copy()

    # Shift to 0-based
    pixel_changes_area['xcoor'] -= x_min
    pixel_changes_area['ycoor'] -= y_min

    prob = np.zeros((tot_pix, num_layers))

    for i in range(len(times_uniq) - 1):
        print(f"Processing time interval {i + 1}/{len(times_uniq) - 1}")
        for k in range(num_layers):
            map_t_ov = comp_pix_tm_map[:, :, i, k]
            flat_ids = map_t_ov.ravel()

            # bincount with +1 offset to handle -1 (unassigned) values
            counts = np.bincount(flat_ids + 1)
            prob[:, k] = counts[flat_ids + 1]**exp
            prob[flat_ids == -1, k] = 1

        tstart = times_uniq[i]
        tend = times_uniq[i + 1]
        t_inds = np.where((pixel_changes_area['seconds'] >= tstart)
                          & (pixel_changes_area['seconds'] < tend))[0]

        prob_sum_ov = np.sum(prob, axis=1)
        sample_prob_pix_time(prob_sum_ov, pixel_changes_area, t_inds)

    # Shift back to real
    pixel_changes_area['xcoor'] += x_min
    pixel_changes_area['ycoor'] += y_min

    return pixel_changes_area


def calc_coords_perimeter(pixel_changes_all, comp_pix_tm_map, times_uniq,
                          coords_comp_time_dict):
    """
    Reassigns pixel coordinates by sampling proportional to the perimeter size
    of compositions. Non-composition pixels are treated as single-pixel
    compositions (weight = 1).

    Parameters
    ----------
    pixel_changes_all : structured ndarray
    comp_pix_tm_map : ndarray, shape (x_dim, y_dim, n_times, num_layers)
    times_uniq : ndarray
    coords_comp_time_dict : dict
    """
    num_layers = comp_pix_tm_map.shape[-1]
    prob = np.zeros((tot_pix, num_layers))
    pixel_changes_perim = pixel_changes_all.copy()

    # Shift to 0-based
    pixel_changes_perim['xcoor'] -= x_min
    pixel_changes_perim['ycoor'] -= y_min

    for t in range(len(times_uniq) - 1):
        print(f"Processing time index {t}/{len(times_uniq) - 2}")
        for k in range(num_layers):
            map_t = comp_pix_tm_map[:, :, t, k]

            # Detect perimeter pixels: composition pixels with at least one
            # neighbor of a different value. Pad with -1 so canvas edges count
            # as perimeter (matching scipy.ndimage.binary_erosion behavior).
            padded = np.pad(map_t, 1, constant_values=-1)
            is_perim = (
                (map_t != padded[:-2, 1:-1]) |
                (map_t != padded[2:, 1:-1]) |
                (map_t != padded[1:-1, :-2]) |
                (map_t != padded[1:-1, 2:])
            ) & (map_t >= 0)

            # Count perimeter pixels per composition
            flat_ids = map_t.ravel()
            perim_counts = np.bincount(
                flat_ids[is_perim.ravel()] + 1,
                minlength=flat_ids.max() + 2)

            prob[:, k] = perim_counts[flat_ids + 1]
            prob[flat_ids == -1, k] = 1

        tstart = times_uniq[t]
        tend = times_uniq[t + 1]
        t_inds = np.where((pixel_changes_perim['seconds'] >= tstart)
                          & (pixel_changes_perim['seconds'] < tend))[0]
        prob_sum_ov = np.sum(prob, axis=1)
        sample_prob_pix_time(prob_sum_ov, pixel_changes_perim, t_inds)

    # Shift back to real
    pixel_changes_perim['xcoor'] += x_min
    pixel_changes_perim['ycoor'] += y_min

    return pixel_changes_perim


def calc_coords_mean_width(pixel_changes_all, comp_pix_tm_map,
                           coords_comp_time_dict, times_uniq):
    """
    Reassigns pixel coordinates by sampling proportional to the mean width
    of the convex hull of each composition (proportional to convex hull
    perimeter, since mean width = perimeter / pi and the constant pi factor
    cancels during normalization). Non-composition pixels get weight 1.

    Parameters
    ----------
    pixel_changes_all : structured ndarray
    comp_pix_tm_map : ndarray, shape (x_dim, y_dim, n_times, num_layers)
    coords_comp_time_dict : dict
    times_uniq : ndarray
    """
    num_layers = comp_pix_tm_map.shape[-1]
    pixel_changes_mw = pixel_changes_all.copy()

    # Shift to 0-based
    pixel_changes_mw['xcoor'] -= x_min
    pixel_changes_mw['ycoor'] -= y_min

    prob = np.zeros((tot_pix, num_layers))

    for i in range(len(times_uniq) - 1):
        print(f"Processing time interval {i + 1}/{len(times_uniq) - 1}")
        for k in range(num_layers):
            flat_ids = comp_pix_tm_map[:, :, i, k].ravel()
            unique_comps = np.unique(flat_ids)
            unique_comps = unique_comps[unique_comps >= 0]

            # Build lookup: mean_widths[comp_id + 1] = convex hull perimeter
            # (proportional to mean width; pi factor omitted as it cancels)
            # Index 0 reserved for -1 (no composition)
            mean_widths = np.zeros(flat_ids.max() + 2)
            for comp_id in unique_comps:
                coords = coords_comp_time_dict[(comp_id, i)]
                n_pixels = coords.shape[1]
                if n_pixels < 3:
                    mean_widths[comp_id + 1] = n_pixels
                    continue
                try:
                    hull = ConvexHull(coords.T)
                    mean_widths[comp_id + 1] = hull.area
                except QhullError:
                    mean_widths[comp_id + 1] = n_pixels

            prob[:, k] = mean_widths[flat_ids + 1]
            prob[flat_ids == -1, k] = 1

        tstart = times_uniq[i]
        tend = times_uniq[i + 1]
        t_inds = np.where((pixel_changes_mw['seconds'] >= tstart)
                          & (pixel_changes_mw['seconds'] < tend))[0]

        prob_sum_ov = np.sum(prob, axis=1)
        sample_prob_pix_time(prob_sum_ov, pixel_changes_mw, t_inds)

    # Shift back to real
    pixel_changes_mw['xcoor'] += x_min
    pixel_changes_mw['ycoor'] += y_min

    return pixel_changes_mw


def calc_coords_loyalty(pixel_changes_all,
                        comp_pix_tm_map,
                        times_uniq,
                        coords_comp_time_dict,
                        seed_real_changes=False):
    """
    Reassigns pixel coordinates to simulate user loyalty behavior.

    Each user's changes are assigned to random pixels within the composition
    of their first pixel change that lands in a known composition. If a user's
    early changes fall on unassigned canvas, the function searches forward
    through their changes until a composition is found.

    Parameters
    ----------
    pixel_changes_all : structured ndarray
    comp_pix_tm_map : ndarray, shape (x_dim, y_dim, n_times, num_layers)
    times_uniq : ndarray
    coords_comp_time_dict : dict
    seed_real_changes : bool
        If True, preserve each user's actual first change.
        If False, randomize the first change.
    """
    num_layers = comp_pix_tm_map.shape[-1]

    pix_changes_user_sorted = np.sort(pixel_changes_all, order='user')

    inds_new_user = np.where(
        np.diff(pix_changes_user_sorted['user']) > 0)[0] + 1
    inds_new_user = np.concatenate(([0], inds_new_user))

    # Randomize first changes BEFORE shifting to 0-based
    # (sample_pixch_quarters operates in real coordinate space)
    if not seed_real_changes:
        pix_changes_user_sorted = sample_pixch_quarters(
            pix_changes_user_sorted, inds_to_sample=inds_new_user)

    # Shift to 0-based
    pix_changes_user_sorted['xcoor'] -= x_min
    pix_changes_user_sorted['ycoor'] -= y_min

    # Precompute time bins and compositions for ALL changes at once
    t_all = np.searchsorted(times_uniq, pix_changes_user_sorted['seconds'], side='right') - 1
    comp_all = comp_pix_tm_map[
        pix_changes_user_sorted['xcoor'],
        pix_changes_user_sorted['ycoor'],
        t_all, 0]
    if num_layers > 1:
        missing = (comp_all == -1)
        comp_all[missing] = comp_pix_tm_map[
            pix_changes_user_sorted['xcoor'][missing],
            pix_changes_user_sorted['ycoor'][missing],
            t_all[missing], 1]

    # Reassign coordinates for each user's changes to random pixels
    # within the composition of their first change that lands in a known composition
    for i, start_change_ind in enumerate(inds_new_user):
        if i % 10000 == 0:
            print('User Number: ' + str(i) + ' out of ' + str(len(inds_new_user)))

        end_change_ind = inds_new_user[i + 1] if i < (len(inds_new_user) - 1) else len(pix_changes_user_sorted)

        user_comps = comp_all[start_change_ind:end_change_ind]
        first_valid = np.argmax(user_comps != -1)

        # argmax returns 0 when no True found, so check the actual value
        if user_comps[first_valid] == -1:
            continue

        compj = user_comps[first_valid]
        tj = t_all[start_change_ind + first_valid]
        idx = start_change_ind + first_valid

        coords_comp_t = coords_comp_time_dict[(compj, tj)]
        num_remaining_changes = end_change_ind - idx
        coords_inds_sampled = np.random.choice(coords_comp_t.shape[1], size=num_remaining_changes)

        pix_changes_user_sorted['xcoor'][idx:end_change_ind] = coords_comp_t[0, coords_inds_sampled]
        pix_changes_user_sorted['ycoor'][idx:end_change_ind] = coords_comp_t[1, coords_inds_sampled]

    # Re-sort by time
    pixel_changes_loyalty = np.sort(pix_changes_user_sorted, order='seconds')

    # Shift back to real
    pixel_changes_loyalty['xcoor'] += x_min
    pixel_changes_loyalty['ycoor'] += y_min

    return pixel_changes_loyalty
