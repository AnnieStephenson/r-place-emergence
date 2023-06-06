import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import shutil
import math
import rplacem.variables_rplace2022 as var
import rplacem.utilities as util
import rplacem.plot_utilities as plot
import rplacem.entropy as entropy
import rplacem.fractal_dim as fractal_dim
import scipy
import warnings
import sparse as sp



def calc_num_pixel_changes(cpart,
                           time_inds_list
                           ):
    '''
    calculate several quantities related to the rate of pixel changes

    parameters
    ----------
    cpart : CanvasPart object
        The CanvasPart object for which we want to calculate the pixel change quantities
    time_inds_list : list of numpy arrays
        list containing arrays of the time indices integrated up to each time step
    time_interval : float
        time interval between frames in seconds

    returns
    -------
    num_pixel_changes : 1d numpy array (length of number of time steps)
        The number of pixel changes in the boundary since the start up to the time interval
    num_touched_pixels : 1d numpy array (length of number of time steps)
        The number of touched pixels (only one change is counted per pixel) since the start
        up to the time interval
    num_users: 1d numpy array (length: number of times steps)
        The number of active user at each time interval
    '''

    num_pixel_changes = np.zeros(len(time_inds_list))
    num_touched_pixels = np.zeros(len(time_inds_list))
    num_users = np.zeros(len(time_inds_list))
    coords_all = cpart.pixchanges_coords()

    for i in range(0, len(time_inds_list)):

        # get number of pixel changes that have had at least one change
        # since the start

        # get the pixel change coordinates for the interval
        pixel_changes_time_int = cpart.pixel_changes[time_inds_list[i]]
        coords = coords_all[:, time_inds_list[i]]
        user_id = np.array(pixel_changes_time_int['user'])

        # get rid of the duplicate pixel changes
        unique_pixel_changes = np.unique(coords, axis=1)
        num_touched_pixels[i] = unique_pixel_changes.shape[1]

        # get the number of unique user ids
        unique_ids = np.unique(user_id)
        num_users[i] = unique_ids.shape[0]

        # number of pixel changes within the current time interval
        num_pixel_changes[i] = len(time_inds_list[i])

    return (num_pixel_changes,
            num_touched_pixels,
            num_users)


def count_image_differences(pixels1, pixels2, cpart, indices=None):
    ''' Count the number of pixels (at given *indices* of coordinates of cpart *cpart*) that differ
    between *pixels1* and *pixels2* (both 2d numpy arrays of shape (num y coords, num x coords))'''
    if indices is None:
        indices = np.arange(0, cpart.num_pix())
    coords = cpart.coords_offset()[:, indices]
    return np.count_nonzero(pixels2[coords[1], coords[0]] - pixels1[coords[1], coords[0]])


def initialize_start_time_grid(cpart, t_start, add_color_dim=False):
    '''
    Initialize to the start of the time interval.
    We neglect the time before the supplementary canvas
    quarters open by setting last_time_changed for those coordinates
    to either the interval start time or the time the coordinates
    become available, whichever happens later.
    '''
    if add_color_dim:
        res = np.full((var.NUM_COLORS, cpart.num_pix()), t_start, dtype='float64')
        res[:, cpart.quarter2_coordinds] = max(t_start, var.TIME_ENLARGE1)
        res[:, cpart.quarter34_coordinds] = max(t_start, var.TIME_ENLARGE2)
    else:
        res = np.full(cpart.num_pix(), t_start, dtype='float64')
        res[cpart.quarter2_coordinds] = max(t_start, var.TIME_ENLARGE1)
        res[cpart.quarter34_coordinds] = max(t_start, var.TIME_ENLARGE2)
    return res


def calc_time_spent_in_color(cpart, seconds, color, pixch_coord_inds,
                             t_beg, t_end, current_color,
                             last_time_installed_sw=None, last_time_removed_sw=None):
    '''
    Returns np.array of shape (n_pixels, n_colors)
    For each timestep, it contains the time that each pixel spent in each color
    Also updates last_time_installed_sw pointer.
    '''

    # Get variables from canvas_part and set shorter names
    coord_range = np.arange(0, cpart.num_pix())
    time_spent_in_color = np.zeros((cpart.num_pix(), var.NUM_COLORS), dtype='float64')

    # Initialize last_time_changed
    last_time_changed = initialize_start_time_grid(cpart, t_beg, add_color_dim=False)

    # Loop through each pixel change in the step, indexing by time.
    for s, c, coor_idx in zip(seconds, color, pixch_coord_inds):
        # nothing necessary here if the color is not changed (redundant pixel change)
        if c == current_color[coor_idx]:
            continue

        # Add the time that this pixel spent in the most recent color.
        time_spent_in_color[coor_idx, current_color[coor_idx]] += s - last_time_changed[coor_idx]

        # Update the time of the last pixel change for this pixel.
        # This code updates the pointer last_time)installed_sw without returning.
        last_time_changed[coor_idx] = s
        if last_time_installed_sw is not None:
            last_time_installed_sw[c][coor_idx] = s # time where this pixel was changed to new color [c]
            last_time_removed_sw[current_color[coor_idx]][coor_idx] = s  # time where this pixel lost its old color [current_color]

        # Update the current color.
        current_color[coor_idx] = c

    # Add the time spent in the final color (from the last pixel change to the end-time).
    time_spent_in_color[coord_range, current_color] += np.maximum(t_end - last_time_changed, 0)

    return time_spent_in_color


def cumulative_attack_timefrac(time_spent_in_col, ref_color, inds_active_pix, stepwidth):
    '''
    From the 2D array time_spent_in_col (shape (n_pixels, n_colors)) and the 1D image ref_color,
    computes the sum of the times that every pixel spent in another color than that of ref_color.
    Normalized by the total time summed over all pixels
    '''
    if len(inds_active_pix) == 0:
        res = 0
    else:
        time_spent = time_spent_in_col[inds_active_pix, :]
        ref_cols = ref_color[inds_active_pix]
        time_spent_in_refcol = time_spent[ np.arange(0, time_spent.shape[0]), ref_cols]

        res = 1 - np.sum(time_spent_in_refcol) / (stepwidth * len(inds_active_pix))
    return max(res, 0)


def calc_stability(stable_timefrac, inds_active, compute_average):
    stab_per_pixel = stable_timefrac[:, 0]  # Get time fraction for most stable color.
    inds_nonzero = np.where(stab_per_pixel > 1e-10)  # Remove indices with stab <= 0.

    # Only use coordinates for which the interval intersects with the 'active' timerange for the composition.
    stab_per_pixel = stab_per_pixel[np.intersect1d(inds_nonzero, inds_active, assume_unique=True).astype(int)]

    # Now get stability averaged over active pixels.
    if compute_average:
        stability = np.mean(stab_per_pixel) if len(stab_per_pixel) > 0 else 1
    else:
        stability = stab_per_pixel

    return stability


def calc_stable_cols(time_spent_in_col):
    return np.flip(np.argsort(time_spent_in_col, axis=1), axis=1)

def substract_and_sparse_replace(a_sw, a_vst, a_replacement, replace=True):
    '''
    modifies [a_sw] into [a_sw - a_vst], where [a_sw] is 1d numpy array and [a_vst] is 1d coo sparse array
    then modifies [a_vst] by a sparse version of [a_replacement]
    '''
    a_sw[a_vst[0]] -= a_vst[1]
    if replace:
        a_vst[0] = np.where(a_replacement != 0)[0]
        a_vst[1] = a_replacement[a_vst[0]]

def show_progress(timefrac, timeflag, frequency=0.1):
    if timefrac > timeflag * frequency:
        timeflag += 1
        print('Ran {:.2f}% of the steps'.format(100 * timefrac), end='\r')
    return 1

def calc_stable_timefrac(cpart, t_step, t_lims,
                         time_spent_in_color, stable_colors):
        '''
        Compute the fraction of time that each pixel was in each color.
        The output stable_timefrac is of shape (n_pixels, n_colors), containing time fractions
        time_spent_in_color is of shape (n_pixels, n_colors), containing spent times (within the given timestep)
        stable_colors is of shape (n_pixels, n_colors), containing color indices ordered in decreasing occupation times
        '''

        res = np.take_along_axis(time_spent_in_color, stable_colors, axis=1)
        # Normalize by the total time the canvas quarter was on.
        res[cpart.quarter1_coordinds] /= t_lims[t_step] - t_lims[t_step-1]
        res[cpart.quarter2_coordinds] /= t_lims[t_step] - max(t_lims[t_step-1], var.TIME_ENLARGE1)
        res[cpart.quarter34_coordinds] /= t_lims[t_step] - max(t_lims[t_step-1], var.TIME_ENLARGE2)

        return res


def main_variables(cpart,
                   cpst,
                   start_pixels=None,
                   delete_dir=False,
                   print_progress=False,
                   flattening='hilbert_pkg',
                   compression='LZ77',
                   save_memory=None
                   ):
    '''
    Calculates the main variables that are contained in a CanvasPartStatistics object.
    This includes stability and attack/defense related variables.

    parameters
    ----------
    cpart : CanvasPart object
        The CanvasPart object for which we want to calculate the stability
    cpst : CanvasPartStatistics object
        The object whose attributes will receive all computed variables
    start_pixels : 1d array, optional, size (n_pixels)
        1D pixels to start with if t0!=0
    delete_dir : boolean, optional
        Remove directory after running, if it did not exist before
    print_progress : boolean, optional
        If True, prints a progress statement
    save_memory : boolean (by default, it is None and then initialized within the function)
        If True, an approach about 2.5x slower is used to store time_spent_in_color using much less RAM

    returns
    -------
    returns nothing, but modifies many attributes of the input cpst
    '''

    # Preliminaries
    t_lims = cpst.t_lims
    n_tlims = len(t_lims)
    attdef = cpst.compute_vars['attackdefense']
    stab = cpst.compute_vars['stability']
    instant = cpst.compute_vars['entropy']
    tran = cpst.compute_vars['transitions']
    other = cpst.compute_vars['other']
    if save_memory is None:
        save_memory = (cpart.num_pix() > 3e5) # do the slower memory-saving method when the canvaspart is larger than 300,000 pixels

    # Some warnings for required conditions of use of the function
    if (not np.isclose(t_lims[0], cpart.minimum_time(), atol=1e-4)) and start_pixels is None:
        warnings.warn('t_lims parameter should start with the minimum time of the CanvasPart, or the start_pixels should be provided, for standard functioning.')
    if delete_dir and (instant > 2 or stab > 2 or attdef > 2):
        warnings.warn('Some images were required to be stored, but with delete_dir=True, the full directory will be removed! ')

    # Canvas part data
    cpart.set_quarters_coord_inds()
    seconds = cpart.pixel_changes['seconds']  # add np.array()?
    color = cpart.pixel_changes['color']
    user = cpart.pixel_changes['user']
    moderator = cpart.pixel_changes['moderator']
    cooldown_cheat = cpart.pixel_changes['cooldown_cheat']
    redundant_color = cpart.pixel_changes['redundant_col']
    redundant_coloranduser = cpart.pixel_changes['redundant_colanduser']
    pixch_coord_inds = cpart.pixel_changes['coord_index']  # indices of cpart.coords where to find the (x,y) of a given pixel_change
    pixch_2Dcoor_offset = cpart.pixchanges_coords_offset()  # for creation of the 2D images
    coor_offset = cpart.coords_offset()  # for transformation from 1D to 2D pixels

    # Initialize variables for first time iteration
    ref_color = cpart.white_image(1)  # reference image (stable over the sliding window)
    current_color = cpart.white_image(1) if start_pixels is None else start_pixels
    if instant > 0 or tran > 0:
        previous_colors = cpart.white_image(2, images_number=cpst.sw_width) # image in the sw_width previous timesteps
    previous_stable_color = cpart.white_image(1)  # stable image in the previous timestep
    if save_memory:
        time_spent_in_color_vst = np.empty((cpst.sw_width, cpart.num_pix(), 2), dtype=object)
        for i in np.ndindex(time_spent_in_color_vst.shape):
            time_spent_in_color_vst[i] = []
    else:
        time_spent_in_color_vst = np.zeros((cpst.sw_width, cpart.num_pix(), var.NUM_COLORS))
    time_spent_in_color_sw = np.zeros((cpart.num_pix(), var.NUM_COLORS)) # summed over the sliding window
    last_time_installed_sw = None if attdef == 0 else initialize_start_time_grid(cpart, t_lims[0], add_color_dim=True)
    last_time_removed_sw = None if attdef == 0 else np.copy(last_time_installed_sw)
    users_vst = np.empty(cpst.sw_width, dtype=object)
    for i in np.ndindex(users_vst.shape):
        users_vst[i] = np.array([], dtype=np.int32)
    users_sw_unique = np.array([], dtype=np.int32)

    # Output
    cpst.stability = cpst.ts_init(np.full(n_tlims, 1., dtype=np.float32))
    cpst.diff_pixels_stable_vs_swref = cpst.ts_init(np.zeros(n_tlims))
    cpst.diff_pixels_inst_vs_swref = cpst.ts_init(np.zeros(n_tlims))
    cpst.diff_pixels_inst_vs_swref_forwardlook = cpst.ts_init(np.zeros(n_tlims))
    cpst.diff_pixels_inst_vs_inst = cpst.ts_init(np.zeros(n_tlims))
    cpst.diff_pixels_inst_vs_stable = cpst.ts_init(np.zeros(n_tlims))
    cpst.returntime = np.zeros((n_tlims, cpart.num_pix()), dtype='float64')
    cpst.area_vst = cpst.ts_init(np.full(n_tlims, cpart.num_pix()))
    cpst.n_changes = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_defense_changes = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_users = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_bothattdef_users = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_defense_users = cpst.ts_init(np.zeros(n_tlims))
    cpst.frac_attack_changes_image = np.full((n_tlims, cpart.width(1), cpart.width(0)), 1, dtype=np.float16) if attdef > 1 else None
    cpst.size_uncompressed = cpst.ts_init(np.zeros(n_tlims))
    cpst.size_compressed = cpst.ts_init(np.zeros(n_tlims))
    cpst.cumul_attack_timefrac = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_moderator_changes = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_cooldowncheat_changes = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_redundant_color_changes = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_redundant_coloranduser_changes = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_users_new_vs_previoustime = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_users_new_vs_sw = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_users_sw = cpst.ts_init( np.zeros(n_tlims) )

    # output paths
    out_path = os.path.join(var.FIGS_PATH, cpart.out_name())
    dir_exist_already = util.make_dir(out_path, renew=False)
    out_dir_time = 'VsTimeInst'
    out_dir_ref = 'reference_image'
    out_dir_stab = 'VsTimeStab'
    out_dir_attdef = 'attack_defense_ratio'
    out_dir_attdefIsing = 'attack_defense_Ising'
    cpart_dir = lambda d : os.path.join(cpart.out_name(), d)
    fig_cpart_dir = lambda d : os.path.join(var.FIGS_PATH, cpart_dir(d))

    # Initialize stored pixels and images
    if stab > 1:
        # 2d numpy arrays containing color indices (from 0 to 31) for each pixel of the composition
        cpst.stable_image = cpart.white_image(3, images_number=n_tlims)
        cpst.second_stable_image = cpart.white_image(3, images_number=n_tlims)
        cpst.third_stable_image = cpart.white_image(3, images_number=n_tlims)

        def modify_some_pixels(start, target, step, indices):
            ''' start and target are 1d arrays of 2d images.
            Coordinates at given indices in [start] must be replaced by the content of [target]'''
            start[step, coor_offset[1, indices], coor_offset[0, indices]] = target[step, coor_offset[1, indices], coor_offset[0, indices]]

        if stab > 2:
            util.make_dir(fig_cpart_dir(out_dir_stab), renew=True)

    if tran > 0:
        cpst.refimage_sw_flat = cpart.white_image(2, images_number=n_tlims)

    if attdef > 1 or tran > 1:
        cpst.refimage_sw = cpart.white_image(3, images_number=n_tlims)
        cpst.attack_defense_image = cpart.white_image(3, images_number=n_tlims)
        if attdef > 2:
            util.make_dir(fig_cpart_dir(out_dir_attdef), renew=True)
            util.make_dir(fig_cpart_dir(out_dir_attdefIsing), renew=True)

    if instant > 0 or tran > 1:
        cpst.true_image = cpart.white_image(3, images_number=n_tlims)
        if instant > 2:
            util.make_dir(fig_cpart_dir(out_dir_time), renew=True)

    # Start with a white image for time time 0
    if ((compression == 'DEFLATE_BMP_PNG') or (instant > 2)):
        if (compression == 'DEFLATE_BMP_PNG') and (flattening != 'ravel'):
            warnings.warn(('Compression algorithm DEFLATE with BMP to PNG can only handle ravel flattening. Using ravel'
                            ' instead of ' + flattening))
        create_files_and_get_sizes(t_lims, 0, cpart.white_image(2),
                                   cpst.size_compressed.val, cpst.size_uncompressed.val, cpart_dir(out_dir_time))

    # LOOP over time steps
    i_fraction_print = 0
    for i in range(1, n_tlims):
        # Print output showing fraction of the steps that have run.
        if print_progress:
            show_progress(i/n_tlims, i_fraction_print, 0.1)
        timerange_str = 'time{:06d}to{:06d}'.format(int(t_lims[i-1]), int(t_lims[i]))

        # Starting time of the sliding window.
        if attdef > 0:
            t_sw_start = t_lims[max(0, i - cpst.sw_width)]
            # Keep only changes within the window (reject older ones).
            last_time_installed_sw = np.maximum(last_time_installed_sw, t_sw_start)
            last_time_removed_sw = np.maximum(last_time_removed_sw, t_sw_start)

        # Get indices of all pixels that are active in this time step.
        inds_coor_active = cpart.active_coord_inds(t_lims[i-1], t_lims[i])
        cpst.area_vst.val[i] = len(inds_coor_active)
        # Get indices of pixel changes in this time step, without or with the condition of being in an "active" pixel at this time
        t_inds = cpart.intimerange_pixchanges_inds(t_lims[i-1], t_lims[i])
        t_inds_active = cpart.select_active_pixchanges_inds(t_inds)

        # CORE COMPUTATIONS. Magic happens here

        # Misc. pixel changes variables
        if other > 0:
            cpst.n_moderator_changes.val[i] = np.count_nonzero(moderator[t_inds_active])
            cpst.n_cooldowncheat_changes.val[i] = np.count_nonzero(cooldown_cheat[t_inds_active])
            cpst.n_redundant_color_changes.val[i] = np.count_nonzero(redundant_color[t_inds_active])
            cpst.n_redundant_coloranduser_changes.val[i] = np.count_nonzero(redundant_coloranduser[t_inds_active])

        # Store the image of the previous timestep
        if stab > 0 and instant > 0:
            previous_stable_color = np.copy(stable_colors[:, 0]) if i > 1 else cpart.white_image(1)

        # Calculate the time each pixel spent in what color.
        # This also updates [current_color, last_time_installed_sw, last_time_removed_sw] with the pixel changes in this time step
        # time_spent_in_color is of shape (n_tlims, n_pixels, n_colors)
        time_spent_in_color = calc_time_spent_in_color(cpart,
                                                       seconds[t_inds], color[t_inds], pixch_coord_inds[t_inds],
                                                       t_lims[i-1], t_lims[i], current_color,
                                                       last_time_installed_sw, last_time_removed_sw)

        # STABILITY
        if stab > 0:
            # Get the color indices in descending order of which color they spent the most time in
            # stable_colors is of shape (n_pixels, n_colors), containing color indices ordered in decreasing occupation times
            stable_colors = calc_stable_cols(time_spent_in_color)

            # Get the times spent in each color in descending order of time spent.
            # stable_timefrac is of shape (n_pixels, n_colors), containing time fractions
            stable_timefrac = calc_stable_timefrac(cpart, i, t_lims,
                                                   time_spent_in_color, stable_colors)
            # calculate the stability value in the time interval
            cpst.stability.val[i] = calc_stability(stable_timefrac, inds_coor_active, True)

        # Store time_spent_in_color for this timestep, in a sparse array
        i_replace = i % cpst.sw_width # where to modify the "rolling" time_spent_in_color_vst array
        if save_memory:
            for j in np.arange(time_spent_in_color_vst.shape[1]):
                substract_and_sparse_replace(time_spent_in_color_sw[j], time_spent_in_color_vst[i_replace, j], time_spent_in_color[j])
        else:
            time_spent_in_color_sw -= time_spent_in_color_vst[i_replace]
            time_spent_in_color_vst[i_replace] = time_spent_in_color
        time_spent_in_color_sw += time_spent_in_color

        # Calculate the new reference image
        ref_color = calc_stable_cols(time_spent_in_color_sw)[:, 0]

        # ATTACK/DEFENSE VS REFERENCE IMAGE
        if attdef > 0:
            # Calculate the (normalized) cumulative attack time over all pixels in this timestep
            cpst.cumul_attack_timefrac.val[i] = cumulative_attack_timefrac(time_spent_in_color, ref_color,
                                                                       inds_coor_active, cpst.t_interval)
            # Calculate return time, as the time each pixel spent in the non-reference color during the last attack
            mask_attack = np.full(current_color.shape, False)
            mask_attack[np.where(current_color != ref_color)] = True # indices where pixels are in an attack color
            cpst.returntime[i][~mask_attack] = last_time_installed_sw[ref_color[~mask_attack], ~mask_attack] \
                                              - last_time_removed_sw[ref_color[~mask_attack], ~mask_attack]
            cpst.returntime[i][mask_attack] = t_lims[i] - last_time_removed_sw[ref_color[mask_attack], mask_attack]

            if attdef > 1:
                diff_image = current_color - ref_color
                diff_image_isref = (diff_image == 0)
                diff_image[diff_image_isref] = 31  # white if defense
                diff_image[np.invert(diff_image_isref)] = 5  # black if attack
                cpst.attack_defense_image[i][coor_offset[1, inds_coor_active],
                                             coor_offset[0, inds_coor_active]] = diff_image[inds_coor_active]

            # Calculate the number of changes and of users that are attacking or defending the reference image
            num_changes_and_users(cpart, i, timerange_str,
                                  user, pixch_coord_inds, pixch_2Dcoor_offset, color,
                                  t_inds_active, ref_color,
                                  cpst.n_changes.val, cpst.n_defense_changes.val,
                                  cpst.n_users.val, cpst.n_bothattdef_users.val, cpst.n_defense_users.val,
                                  cpst.frac_attack_changes_image,
                                  attdef > 1, (cpart_dir(out_dir_attdef) if attdef > 2 else ''))
            
            # Compare number of unique users to that in the previous timestep or sliding window
            users_now = user[t_inds_active]
            i_prev = (i-1) % cpst.sw_width
            users_prev = users_vst[i_prev]
            cpst.n_users_new_vs_previoustime.val[i] = len(np.intersect1d(users_now, users_prev)) # count unique elements in common in the two arrays
            cpst.n_users_new_vs_sw.val[i] = len(np.intersect1d(users_now, users_sw_unique))

            # Number of unique users in a sliding window
            users_vst[i_replace] = users_now
            users_sw_unique = np.unique(np.hstack(users_vst))
            cpst.n_users_sw.val[i] = len(users_sw_unique)

        # INSTANTANEOUS IMAGES: includes entropy and fractal dimension calculations
        # Calculate the number of pixels in the current interval (stable or instantaneous) that differ from the reference image, or from the previous timestep
        if instant > 0 or tran > 0:
            cpst.diff_pixels_inst_vs_swref.val[i] = np.count_nonzero(current_color - ref_color)
        if tran > 0:
            if i >= cpst.sw_width:
                cpst.diff_pixels_inst_vs_swref_forwardlook.val[i - cpst.sw_width] = np.count_nonzero(previous_colors[i_replace] - ref_color)
            previous_colors[i_replace] = np.copy(current_color)
        if stab > 0 and instant > 0:
            cpst.diff_pixels_stable_vs_swref.val[i] = np.count_nonzero(stable_colors[:, 0] - ref_color[:])
            cpst.diff_pixels_inst_vs_stable.val[i] = np.count_nonzero(current_color - previous_stable_color)
        if instant > 0:
            # ENTROPY
            cpst.diff_pixels_inst_vs_inst.val[i] = np.count_nonzero(current_color - previous_colors[(i-1) % cpst.sw_width])

            # Create the png and bmp files from the current image, and store their sizes
            pix_tmp = cpart.white_image(2)
            pix_tmp[coor_offset[1, inds_coor_active], coor_offset[0, inds_coor_active]] = current_color[inds_coor_active]
            if ((compression == 'DEFLATE_BMP_PNG') or (instant > 2)):
                create_files_and_get_sizes(t_lims, i, pix_tmp,
                                           cpst.size_compressed.val, cpst.size_uncompressed.val,
                                           cpart_dir(out_dir_time), True, instant < 3)
            else:
                cpst.size_compressed.val[i] = entropy.calc_compressed_size(pix_tmp, flattening=flattening, compression=compression)
                cpst.size_uncompressed.val[i] = entropy.calc_size(pix_tmp)

        # END CORE COMPUTATIONS. Magic ends

        # Save full result to pickle file.
        if stab > 3:
            file_path = os.path.join(var.DATA_PATH, 'stability_' + cpart.out_name() + '_time{:06d}to{:06d}.pickle'.format(int(t_lims[i-1]), int(t_lims[i])))
            with open(file_path, 'wb') as handle:
                pickle.dump([stable_timefrac,
                            stable_colors],
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        # create instantaneous images.
        if instant > 0 or tran > 1:
            cpst.true_image[i] = np.copy(pix_tmp)

        # Create attack/defense images.
        if tran > 0:
            cpst.refimage_sw_flat[i] = ref_color
        if attdef > 1 or tran > 1:
            cpst.refimage_sw[i, coor_offset[1], coor_offset[0]] = ref_color
            if attdef > 2:
                timerange_str_ref = 'time{:06d}to{:06d}'.format(int(t_sw_start), int(t_lims[i]))
                util.pixels_to_image(cpst.refimage_sw[i], cpart_dir(out_dir_ref), 'SlidingRef_' + timerange_str_ref + '.png')
                util.pixels_to_image(cpst.attack_defense_image[i], cpart_dir(out_dir_attdefIsing), 'Attack_vs_defense_' + timerange_str + '.png')

        # Create images containing the (sub)dominant color only.
        if stab > 1:
            cpst.stable_image[i, coor_offset[1], coor_offset[0]] = stable_colors[:, 0]
            cpst.second_stable_image[i, coor_offset[1], coor_offset[0]] = stable_colors[:, 1]
            cpst.third_stable_image[i, coor_offset[1], coor_offset[0]] = stable_colors[:, 2]

            # If second and/or third most used colors don't exist (time_spent == 0),
            # then use the first or second most used color instead.
            inds_to_change1 = np.where(stable_timefrac[:, 1] < 1e-9)
            modify_some_pixels(cpst.second_stable_image, cpst.stable_image, i, inds_to_change1)
            inds_to_change2 = np.where(stable_timefrac[:, 2] < 1e-9)
            modify_some_pixels(cpst.third_stable_image, cpst.second_stable_image, i, inds_to_change2)

            # Save images.
            if stab > 2:
                util.pixels_to_image(cpst.stable_image[i], cpart_dir(out_dir_stab), 'MostStableColor_' + timerange_str + '.png')
                util.pixels_to_image(cpst.second_stable_image[i], cpart_dir(out_dir_stab), 'SecondMostStableColor_' + timerange_str + '.png')
                util.pixels_to_image(cpst.third_stable_image[i], cpart_dir(out_dir_stab), 'ThirdMostStableColor_' + timerange_str + '.png')

    # FRACTAL DIMENSION
    if instant > 0:
        [cpst.fractal_dim_mask_median.val,
         cpst.fractal_dim_weighted.val] = fractal_dim.calc_from_image(cpst.true_image, shift_avg=False)

    # Continue the loop over some more time steps to get the forward-looking sliding window reference
    if tran > 0:
        for i in range(n_tlims, n_tlims + cpst.sw_width):
            i_replace = i % cpst.sw_width # where to modify the "rolling" time_spent_in_color_vst array
            if save_memory:
                for j in np.arange(time_spent_in_color_vst.shape[1]):
                    substract_and_sparse_replace(time_spent_in_color_sw[j], time_spent_in_color_vst[i_replace, j], None, replace=False)
            else:
                time_spent_in_color_sw -= time_spent_in_color_vst[i_replace]

            ref_color = calc_stable_cols(time_spent_in_color_sw)[:, 0]
            cpst.diff_pixels_inst_vs_swref_forwardlook.val[i - cpst.sw_width] = np.count_nonzero(previous_colors[i_replace] - ref_color)

    # These calculations can be done on the final arrays
    cpst.n_users_total = len(np.unique(user))

    if print_progress:
        print('                              ', end='\r')

    if delete_dir and not dir_exist_already:
        os.rmdir(out_path)


def num_changes_and_users(cpart, t_step, time_str,
                          user, pixch_coord_inds, pixch_2Dcoor_offset, color,
                          t_inds_active, ref_image,
                          num_changes, num_defense_changes,
                          num_users, num_attackdefense_users, num_defenseonly_users,
                          pixels_fracattack,
                          save_ratio_pixels, save_ratio_images=''):
    '''
    Modifies multiple variables dealing with the number of pixel changes and users
    and their relation to the provided reference (stable) image

    parameters
    ----------
    cpart : CanvasPart object
        The CanvasPart object under study
    ref_image: 2d array, shape (# y coords, # of x coords)
        Image to which pixels at every time step are compared
    save_ratio_pixels: bool
        save the pixel arrays of the pixel-by-pixel attack/defense ratio
    save_ratio_images: bool
        save the images of the pixel-by-pixel attack/defense ratio
    num_changes, num_defense_changes:
        the number of total pixel changes, those that restore the ref_image (defense), and those that destroy the ref_image (attack).
    num_active_coords:
        Active area of the composition in each timerange
    num_differing_pixels:
        number of active pixels that are different than the ref image
    num_defenseonly_users, num_attackdefense_users, num_users:
        number of users that contributed changes (defending, both attacking and defending, or all) in the given time interval
    '''

    # get indices of all pixel changes that happen in the step (and that are part of the composition timerange for this pixel)
    num_changes[t_step] = len(t_inds_active)
    num_users[t_step] = len(np.unique(user[t_inds_active]))

    # pixel changes that agree or not with the ref image
    agreeing_changes = np.array(ref_image[pixch_coord_inds[t_inds_active]] == color[t_inds_active], np.bool_)
    disagree_changes = np.invert(agreeing_changes)
    num_defense_changes[t_step] = np.count_nonzero(agreeing_changes)

    # count users making defense or attack moves
    defense_users = np.unique(user[t_inds_active][agreeing_changes])
    attack_users = np.unique(user[t_inds_active][disagree_changes])
    attackdefense_users = np.intersect1d(attack_users, defense_users)
    num_attackdefense_users[t_step] = len(attackdefense_users)
    num_defense_users = len(defense_users)
    num_defenseonly_users[t_step] = num_defense_users - num_attackdefense_users[t_step]

    # count attack and defense changes for each pixel of the canvas
    if save_ratio_pixels:
        xcoords = pixch_2Dcoor_offset[0][t_inds_active]
        ycoords = pixch_2Dcoor_offset[1][t_inds_active]

        att = np.zeros((cpart.width(1), cpart.width(0)), dtype=np.float16)
        defe = np.zeros((cpart.width(1), cpart.width(0)), dtype=np.float16)
        for t in range(0, len(t_inds_active)):
            if agreeing_changes[t]:
                defe[ycoords[t], xcoords[t]] += 1
            else:
                att[ycoords[t], xcoords[t]] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            pixels_fracattack[t_step] = att / (defe + att)
        inds_nan = np.where((att>0) & (defe==0))
        pixels_fracattack[t_step][inds_nan[0], inds_nan[1]] = 100.

        if save_ratio_images != '':
            plt.figure()
            pcm = plt.pcolormesh(np.arange(0,cpart.width()[0]), np.arange(cpart.width()[1] - 1, -1, -1), pixels_fracattack[t_step], shading='nearest')
            plt.xlabel('x_pixel')
            plt.ylabel('y_pixel')
            plt.colorbar(pcm, label='# attack / # defense changes')
            plt.clim(0.99, 1.01)
            plt.savefig(os.path.join(var.FIGS_PATH, save_ratio_images, 'attack_defense_ratio_perpixel_'+time_str))
            plt.close()

    return 1


def create_files_and_get_sizes(t_lims, t_step, pixels,
                               file_size_png, file_size_bmp,
                               out_path_time, delete_bmp=True, delete_png=False):
    '''
    Creates bmp and png images from the 2D [pixels].
    Stores them, and records their sizes into file_size_png and file_size_bmp
    '''
    namecore = 'canvaspart_time{:06d}'.format(int(t_lims[t_step]))
    _, impath_png, impath_bmp = util.pixels_to_image(pixels, out_path_time, namecore + '.png', namecore + '.bmp')

    file_size_png[t_step] = util.get_file_size(impath_png)
    file_size_bmp[t_step] = util.get_file_size(impath_bmp)
    if delete_bmp:
        os.remove(impath_bmp)
    if delete_png:
        os.remove(impath_png)


def return_time_fwdlooking(cpart, ref_image, t_lims=[0, var.TIME_TOTAL], summary_stats=True):
    '''
    Return time for all attack pixel changes: the time that each pixel
    that was changed from its reference color (in ref_image) to return
    to the reference color. Also returns the time at which those pixels wer attacked.
    '''

    # Canvas part data
    sortcoor = cpart.pixch_sortcoord  # sort pixel changes by pixel coordinate
    color = cpart.pixel_changes['color'][sortcoor]
    coordidx = cpart.pixel_changes['coord_index'][sortcoor]
    time = cpart.pixel_changes['seconds'][sortcoor]
    pixchanges_coor_offset = cpart.pixchanges_coords_offset()
    xcoords = pixchanges_coor_offset[0][sortcoor]
    ycoords = pixchanges_coor_offset[1][sortcoor]

    # should be of form [pixel 1 att(False), def(True), att, att, ..., pixel 2 def,def,att,def,att,...]
    agreeing_changes = np.array( ref_image[ycoords, xcoords] == color, np.bool_)
    disagree_changes = np.invert(agreeing_changes)
    # does the pixel change correspond to a new pixel?
    coord_changed = np.hstack(( True, (np.diff(coordidx) > 0) ))
    coord_change_ind = np.where(coord_changed)[0]
    # is this pixel change the beginning of an attack or defense sequence?
    newly_attacked_or_restored = np.hstack((True, np.diff(agreeing_changes)))
    newly_attacked_or_restored[coord_changed & disagree_changes] = True # consider that it is newly attacked when looking at another pixel
    newly_attacked_or_restored[coord_changed & agreeing_changes] = False # defense changes at the start of a new pixel are ignored

    beg_att_seq_ind = np.where(disagree_changes & newly_attacked_or_restored)[0]
    beg_def_seq_ind = np.where(agreeing_changes & newly_attacked_or_restored)[0]
    for i in range(0, len(coord_change_ind) - 1):
        thispix = np.arange(coord_change_ind[i], coord_change_ind[i+1])
        new_att = np.intersect1d(beg_att_seq_ind, thispix, assume_unique=True)
        new_def = np.intersect1d(beg_def_seq_ind, thispix, assume_unique=True)
        if len(new_att) == len(new_def) + 1:
            unrestored_att_ind = new_att[-1]
            new_att = new_att[:-1]
            returntime += [var.TIME_TOTAL]
            time_attackflip += [time[unrestored_att_ind]]
        returntime += list(time[new_def] - time[new_att])
        time_attackflip += list(time[new_att])
    returntime = np.array(returntime)
    time_attackflip = np.array(time_attackflip)

    returntime_tbinned = None
    returntime_mean = None
    returntime_median_overln2 = None
    if summary_stats:
        returntime_tbinned = np.empty(len(t_lims)-1, dtype=object)
        returntime_mean = np.empty(len(t_lims)-1)
        returntime_median_overln2 = np.empty(len(t_lims)-1)
        timebin_ind = np.digitize(time_attackflip, t_lims) - 2  # TODO: check this
        for j in range(0, len(t_lims)-1):  # initialize with empty lists
            returntime_tbinned[j] = []
        for j in range(0, len(returntime)):  # fill the histogram-like array using the result from np.digitize
            returntime_tbinned[timebin_ind[j]].append(returntime[j])
        for j in range(0, len(t_lims)-1):  # cannot use more clever numpy because the lists in axis #3 are of different sizes
            returntime_mean[j] = np.mean(np.array(returntime_tbinned[j]))
            returntime_median_overln2[j] = np.median(np.array(returntime_tbinned[j]))
        returntime_median_overln2 /= np.log(2)

    return (returntime, time_attackflip, returntime_tbinned, returntime_mean, returntime_median_overln2)
