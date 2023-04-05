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
import scipy


def calc_num_pixel_changes(cpart,
                           time_inds_list,
                           time_interval):
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


def stability(cpart,
              t_lims=[0, var.TIME_TOTAL],
              create_images=False,
              save_images=False,
              save_pickle=False,
              compute_average=True,
              print_progress=True,
              sliding_window_time=None,
              prev_stab_col=None,
              t_unit=300
              ):
    '''
    makes map of stability in some time range.
    Stability of a pixel is the fraction of time it spent in its 'favorite' color (meaning the color it was in for the most time)

    parameters
    ----------
    cpart : CanvasPart object
        The CanvasPart object for which we want to calculate the stability
    t_lims : 1d array-like of floats
        time intervals in which the pixel states are studied. Must be ordered in a crescent way, and start at 0
    create_images, save_images, save_pickle, compute_average: bools
        flags, see 'returns' section

    returns
    -------
    stability_vs_time: 1d array-like of floats
        stability array averaged over all the pixels, for each time step
        Calculated only if compute_average==True
    The saved pickle file (if save_pickle==True) records the stable_timefraction (time fraction spent in the dominant color)
        and the stable_color (the dominant colors). They are indexed in the same way than cpart.coords.
    The images (computed if create_images==True, saved if save_images==True) show
        the most stable color of each pixel during each time interval.
    '''

    if t_lims[0] != 0:
        print('WARNING: this function \'stability\' is not meant to work from a lower time limit > 0 !!!')

    # Get variables from canvas_part and set shorter names
    seconds = np.array(cpart.pixel_changes['seconds'])
    color = np.array(cpart.pixel_changes['color'])
    pixch_coord_inds = np.array(cpart.pixel_changes['coord_index'])  # indices of cpart.coords where to find the (x,y) of a given pixel_change
    coord_range = np.arange(0, cpart.num_pix())
    cpart.set_quarters_coord_inds()

    # Initialize variables for first iteration
    current_color = cpart.white_image(1)
    step_start_color = cpart.white_image(1)
    previous_stable_colors = cpart.white_image(1)
    stability_vs_time = np.zeros(len(t_lims)-1)
    instab_vs_time_norm = np.zeros(len(t_lims)-1)
    diff_pixels_vst = np.zeros(len(t_lims)-1)
    coor_offset = cpart.coords_offset()

    if create_images:
        util.make_dir(os.path.join(var.FIGS_PATH, cpart.out_name()))
        util.make_dir(os.path.join(var.FIGS_PATH, cpart.out_name(),
                                   'VsTimeStab'))

        # 2d numpy arrays containing color indices (from 0 to 31) for each pixel of the composition
        pixels1 = cpart.white_image(3, images_number=len(t_lims) - 1)
        pixels2 = cpart.white_image(3, images_number=len(t_lims) - 1)
        pixels3 = cpart.white_image(3, images_number=len(t_lims) - 1)

        def modify_some_pixels(start, target, step, indices):
            ''' start and target are 1d arrays of 2d images.
            Coordinates at given indices in [start] must be replaced by the content of [target]'''
            start[step, coor_offset[1, indices], coor_offset[0, indices]] = target[step, coor_offset[1, indices], coor_offset[0, indices]]

    i_fraction_print = 0
    for i in range(len(t_lims) - 1):
        # Print output message to show fraction of the steps that have run.
        if print_progress:
            if i/len(t_lims) > i_fraction_print / 10:
                i_fraction_print += 1
                print('Ran {:.2f}% of the steps'.format(100 * i / len(t_lims)), end='\r')

        # Get indices of all pixel changes that happen in the step.
        if sliding_window_time is not None:
            # this is currently a forward-looking time bin.
            # in the future, we want this to be backward-looking
            t_win_start = t_lims[i]
            if t_win_start < 1:
                t_win_start = 0
            t_win_end = t_lims[i] + sliding_window_time
            if t_win_end > t_lims[-1]:
                t_win_end = t_lims[-1]
        else:
            t_win_start = t_lims[i]  # need to shift for loop
            t_win_end = t_lims[i+1]
        t_inds = cpart.intimerange_pixchanges_inds(t_win_start, t_win_end)

        # Time spent for each pixel in each of the 32 colors
        time_spent_in_color = np.zeros((cpart.num_pix(), var.NUM_COLORS), dtype='float64')

        # Define the last_time_changed, initializing to the start of the time
        # interval. We negect the time before the supplementary canvas
        # quarters open by setting last_time_changed for those coordinates
        # to either the interval start time or the time the coordinates
        # become available, whichever happens later.
        last_time_changed = np.full(cpart.num_pix(), t_win_start, dtype='float64')
        last_time_changed[cpart.quarter2_coordinds] = max(t_win_start, var.TIME_ENLARGE1)
        last_time_changed[cpart.quarter34_coordinds] = max(t_win_start, var.TIME_ENLARGE2)

        time_flag = False  # Flag marks when start time of next window is reached.

        # Loop through each pixel change in the step, indexing by time.
        for j, tidx in enumerate(t_inds):
            s = seconds[tidx]
            c = color[tidx]
            coor_idx = pixch_coord_inds[tidx]

            # If this is the first iteration in the loop,
            # then set the current color to the step_start_color.
            # This enables overlapping windows to have correct starting colors.
            if j == 0:
                current_color = step_start_color.copy()

            # Add the time that this pixel spent in the most recent color.
            time_spent_in_color[coor_idx, current_color[coor_idx]] += s - last_time_changed[coor_idx]

            # Update the time of the last pixel change for this pixel.
            last_time_changed[coor_idx] = s

            # If the pixel change is past the start time of the next window
            # then set step_start_color to current_color, to use as the
            # initial current_color at the start of the next window.
            if s > (t_lims[i + 1]) and not time_flag:
                step_start_color = current_color.copy()
                time_flag = True

            # Update the current color.
            current_color[coor_idx] = c

        # Add the time spent in the final color (from the last pixel change to the end-time).
        time_spent_in_color[coord_range, current_color] += np.maximum(t_win_end - last_time_changed, 0)

        # Find coordinates where there are no pixel changes
        # and add the entire time to them.
        no_change_coords = np.where(~time_spent_in_color.any(axis=1))[0]
        if len(no_change_coords) > 0:
            no_change_colors = current_color[no_change_coords]
            time_spent_in_color[no_change_coords, no_change_colors] += (t_win_end - t_win_start)

        # Get the color indices in descending order of which color they spent the most time in
        stable_colors = np.flip(np.argsort(time_spent_in_color, axis=1), axis=1)

        # Get the times spent in each color in descending order of time spent.
        stable_timefraction = np.take_along_axis(time_spent_in_color, stable_colors, axis=1)

        # Normalize by the total time the canvas quarter was on.
        if t_lims[i] < t_lims[i + 1]:
            stable_timefraction[cpart.quarter1_coordinds] /= t_lims[i+1] - t_lims[i]
            stable_timefraction[cpart.quarter2_coordinds] /= t_lims[i+1] - max(t_lims[i], var.TIME_ENLARGE1)
            stable_timefraction[cpart.quarter34_coordinds] /= t_lims[i+1] - max(t_lims[i], var.TIME_ENLARGE2)

        # Calculate the stability averaged over the whole cpart.
        if compute_average:
            stab_per_pixel = stable_timefraction[:, 0]  # Get time fraction for most stable pixel.
            inds_nonzero = np.where(stab_per_pixel > 1e-10)  # Remove indices with stab == 0.

            # Get indices of coordinates for which the interval [t_lims[t_step], t_lims[t_step+1]] intersects with the 'active' timerange for the composition.
            inds_active = cpart.active_coord_inds(t_lims[i], t_lims[i + 1])
            stab_per_pixel = stab_per_pixel[np.intersect1d(inds_nonzero, inds_active, assume_unique=True).astype(int)]

            # Now get stability averaged over active pixels.
            if len(stab_per_pixel) > 0:
                stability = np.mean(stab_per_pixel)
            else:
                stability = 1
            stability_vs_time[i] = stability

        # Save full result to pickle file.
        if save_pickle:
            file_path = os.path.join(var.DATA_PATH, 'stability_' + cpart.out_name() + '_time{:06d}to{:06d}.pickle'.format(int(t_win_start), int(t_win_end)))
            with open(file_path, 'wb') as handle:
                pickle.dump([stable_timefraction,
                            stable_colors],
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        diff_pixels_vst[i] = np.count_nonzero(stable_colors[:, 0] - previous_stable_colors[:])

        if prev_stab_col is not None:
            previous_stable_colors = prev_stab_col[i - 1, coor_offset[1], coor_offset[0]]
        else:
            previous_stable_colors = stable_colors[:, 0]

        # Create images containing the (sub)dominant color only.
        if create_images:
            pixels1[i, coor_offset[1], coor_offset[0]] = stable_colors[:, 0]
            pixels2[i, coor_offset[1], coor_offset[0]] = stable_colors[:, 1]
            pixels3[i, coor_offset[1], coor_offset[0]] = stable_colors[:, 2]

            # If second and/or third most used colors don't exist (time_spent == 0),
            # then use the first or second most used color instead.
            inds_to_change1 = np.where(stable_timefraction[:, 1] < 1e-9)
            modify_some_pixels(pixels2, pixels1, i, inds_to_change1)
            inds_to_change2 = np.where(stable_timefraction[:, 2] < 1e-9)
            modify_some_pixels(pixels3, pixels2, i, inds_to_change2)

            # Save images.
            if save_images:
                timerange_str = 'time{:06d}to{:06d}.png'.format(int(t_win_start), int(t_win_end))
                util.pixels_to_image(pixels1[i], os.path.join(cpart.out_name(), 'VsTimeStab'), 'MostStableColor_' + timerange_str + '.png')
                util.pixels_to_image(pixels2[i], os.path.join(cpart.out_name(), 'VsTimeStab'), 'SecondMostStableColor_' + timerange_str + '.png')
                util.pixels_to_image(pixels3[i], os.path.join(cpart.out_name(), 'VsTimeStab'), 'ThirdMostStableColor_' + timerange_str + '.png')

    t_interval = np.diff(t_lims)
    with np.errstate(divide='ignore', invalid='ignore'):
        instab_vs_time_norm = (1 - stability_vs_time) * t_unit / t_interval
    instab_vs_time_norm[np.where(t_interval == 0)] = 0

    if print_progress:
        print('                              ', end='\r')
    if not compute_average:
        stability_vs_time = np.array([])
    if not create_images:
        (pixels1, pixels2, pixels3) = (np.array([]), np.array([]), np.array([]))

    return [stability_vs_time, instab_vs_time_norm, pixels1, pixels2, pixels3, diff_pixels_vst]


def num_changes_and_users(cpart, t_lims, ref_image, save_ratio_pixels,
                          save_ratio_images, simple_notrans=False):
    '''
    Returnes multiple variables dealing with the number of pixel changes and users
    and their relation to the provided reference (stable) image

    parameters
    ----------
    cpart : CanvasPart object
        The CanvasPart object under study
    t_lims : 1d array-like of floats
        time intervals in which the pixel states are studied. Must be ordered in a crescent way, and start at 0
    ref_image: 2d array, shape (# y coords, # of x coords)
        Image to which pixels at every time step are compared
    save_ratio_pixels: bool
        save the pixel arrays of the pixel-by-pixel attack/defense ratio
    save_ratio_images: bool
        save the images of the pixel-by-pixel attack/defense ratio
    simple_notrans: bool
        only calculate number of pix changes and users, without notions of attack and defense

    returns
    -------
    num_changes, num_attack_changes, num_defense_changes:
        the number of total pixel changes, those that restore the ref_image (defense), and those that destroy the ref_image (attack).
    num_active_coords:
        Active area of the composition in each timerange
    num_differing_pixels:
        number of active pixels that are different than the ref image
    num_attackonly_users, num_defenseonly_users, num_attackdefense_users, num_users_total:
        number of users that contributed changes (attacking, defending, doing both, or all) in the given time interval
    '''

    if t_lims[0] != 0:
        print('WARNING: this function \'num_deviating_pixels\' is not meant to work from a lower time limit > 0 !!!')

    if save_ratio_images:
        util.make_dir(os.path.join(var.FIGS_PATH, cpart.out_name()), renew=False)
        util.make_dir(os.path.join(var.FIGS_PATH, cpart.out_name(), 'attack_defense_ratio'), renew=True)

    sortcoor = cpart.pixch_sortcoord # sort pixel changes by pixel coordinate
    user = cpart.pixel_changes['user'][sortcoor]
    if not simple_notrans:
        color = cpart.pixel_changes['color'][sortcoor]
        coordidx = cpart.pixel_changes['coord_index'][sortcoor]
        time = cpart.pixel_changes['seconds'][sortcoor]
        pixchanges_coor_offset = cpart.pixchanges_coords_offset()
        xcoords = pixchanges_coor_offset[0][sortcoor]
        ycoords = pixchanges_coor_offset[1][sortcoor]

    num_changes = np.zeros(len(t_lims)-1)
    num_users = np.zeros(len(t_lims)-1)
    num_active_pix = np.zeros(len(t_lims)-1)
    if not simple_notrans:
        num_defense_changes = np.zeros(len(t_lims)-1)
        num_differing_pixels = np.zeros(len(t_lims)-1)
        num_attackdefense_users = np.zeros(len(t_lims)-1)
        num_attackonly_users = np.zeros(len(t_lims)-1)
        num_defenseonly_users = np.zeros(len(t_lims)-1)
        returntime = []
        time_attackflip = []
    if save_ratio_pixels:
        pixels = np.zeros((len(t_lims)-1, cpart.width(1), cpart.width(0)), dtype=np.float16)
    else:
        pixels = None

    if not simple_notrans:
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

    current_image = cpart.white_image(2)
    for t_step in range(0, len(t_lims)-1):
        # get indices of all pixel changes that happen in the step (and that are part of the composition timerange for this pixel)
        t_inds = cpart.intimerange_pixchanges_inds(t_lims[t_step], t_lims[t_step+1], sortcoor)
        t_inds_active = cpart.select_active_pixchanges_inds(t_inds, sortcoor)
        num_changes[t_step] = len(t_inds_active)
        num_users[t_step] = len(np.unique(user[t_inds_active]))

        # count active (ie in timerange) coordinates and the differences with the ref_image at these coordinates
        active_coor_inds = cpart.active_coord_inds(t_lims[t_step], t_lims[t_step+1])
        num_active_pix[t_step] = len(active_coor_inds)

        if simple_notrans: continue

        # test if the color of this pixel change agrees with the reference image (ie is it of the same color)
        agreeing_changes_now = agreeing_changes[t_inds_active]
        num_defense_changes[t_step] = np.count_nonzero(agreeing_changes_now)
        agree_indinds = np.where(agreeing_changes_now)[0]
        disagree_indinds = np.where(disagree_changes[t_inds_active])[0]

        # count users making defense or attack moves
        defense_users = np.unique( user[t_inds_active[agree_indinds]] )
        attack_users = np.unique( user[t_inds_active[disagree_indinds]] )
        attackdefense_users = np.intersect1d(attack_users, defense_users)
        num_attackdefense_users[t_step] = len(attackdefense_users)
        num_attack_users = len(attack_users)
        num_defense_users = len(defense_users)
        num_attackonly_users[t_step] = num_attack_users - num_attackdefense_users[t_step]
        num_defenseonly_users[t_step] = num_defense_users - num_attackdefense_users[t_step]

        # count attack and defense changes for each pixel of the canvas
        if save_ratio_pixels:
            att = np.zeros((cpart.width(1), cpart.width(0)), dtype=np.float16)
            defe = np.zeros((cpart.width(1), cpart.width(0)), dtype=np.float16)
            for t in agree_indinds:
                defe[ycoords[t_inds_active[t]], xcoords[t_inds_active[t]]] += 1
            for t in disagree_indinds:
                att[ycoords[t_inds_active[t]], xcoords[t_inds_active[t]]] += 1
            with np.errstate(divide='ignore', invalid='ignore'):
                pixels[t_step] = att / defe
            inds_nan = np.where((att>0) & (defe==0))
            pixels[t_step][inds_nan[0], inds_nan[1]] = 100.

            if save_ratio_images:
                plt.figure()
                pcm = plt.pcolormesh(np.arange(0,cpart.width()[0]), np.arange(cpart.width()[1] - 1, -1, -1), pixels[t_step], shading='nearest')
                plt.xlabel('x_pixel')
                plt.ylabel('y_pixel')
                plt.colorbar(pcm, label='# attack / # defense changes')
                plt.clim(0.99,1.01)
                outpath = os.path.join(var.FIGS_PATH, cpart.out_name(), 'attack_defense_ratio')
                util.make_dir(outpath)
                plt.savefig(os.path.join(outpath, 'attack_defense_ratio_perpixel_time{:06d}.png'.format(int(t_lims[t_step+1]))))
                plt.close()

        # Update current_image with the pixel changes in this time interval.
        util.update_image(current_image, xcoords, ycoords, color, t_inds)

        num_differing_pixels[t_step] = count_image_differences(current_image, ref_image, cpart, active_coor_inds) if num_active_pix[t_step] != 0 else 0

    if not simple_notrans:
        num_attack_changes = num_changes - num_defense_changes
    num_users_total = len(np.unique(user))

    return (num_changes,
            num_defense_changes if not simple_notrans else None,
            num_attack_changes if not simple_notrans else None,
            num_active_pix,
            num_differing_pixels if not simple_notrans else None,
            num_attackonly_users if not simple_notrans else None,
            num_defenseonly_users if not simple_notrans else None,
            num_attackdefense_users if not simple_notrans else None,
            num_users, num_users_total,
            pixels if not simple_notrans else None,
            returntime if not simple_notrans else None,
            time_attackflip if not simple_notrans else None)

def save_part_over_time(cpart,
                        times, # in seconds
                        record_pixels=False,
                        delete_bmp=True,
                        delete_png=False,
                        show_plot=True,
                        print_progress=False,
                        remove_inactive=True
                        ):
    '''
    Saves images of the canvas part for each time step

    parameters
    ----------
    cpart : CanvasPart object
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

    if cpart.is_rectangle and remove_inactive:
        remove_inactive = False # no need of removing inactive (vs time) pixels in this case
    act_coor_inds = np.arange(cpart.num_pix())

    pixchanges_coor_offset = cpart.pixchanges_coords_offset()
    xcoords = pixchanges_coor_offset[0]
    ycoords = pixchanges_coor_offset[1]
    color = np.array(cpart.pixel_changes['color'])

    num_time_steps = len(times)-1
    file_size_bmp = np.zeros(num_time_steps+1)
    file_size_png = np.zeros(num_time_steps+1)
    num_active_pix = np.zeros(num_time_steps+1)
    num_differing_pixels = np.zeros(num_time_steps+1)

    if record_pixels:
        # 2d numpy arrays containing color indices for each pixel of the composition, at each time step
        pixels_vst = cpart.white_image(3, images_number=num_time_steps+1)
    else:
        pixels_vst = None

    pixels_previous = cpart.white_image(2)
    pixels = cpart.white_image(2) # fill as white first # the pixels must be [y,x,rgb]
    out_path = os.path.join(var.FIGS_PATH, cpart.out_name())
    out_path_time = os.path.join(out_path, 'VsTime')
    util.make_dir(out_path)
    util.make_dir(out_path_time)

    if show_plot:
        ncols = np.min([num_time_steps, 10])
        nrows = np.max([1, int(math.ceil(num_time_steps/10))])
        fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        rowcount = 0
        colcount = 0
    t_inds_list = []

    i_fraction_print = 0  # only for output of a message when some fraction of the steps are ran

    for t_step in range(0, num_time_steps+1):  # fixed this: want a blank image at t=0
        if print_progress:
            if t_step/num_time_steps > i_fraction_print/10:
                i_fraction_print += 1
                print('Ran {:.2f}% of the steps'.format(100*t_step/num_time_steps), end='\r')

        # get the indices of the times within the interval
        t_inds = cpart.intimerange_pixchanges_inds(times[t_step - 1], times[t_step]) # does NOT exclude inactive pixels
        t_inds_list.append(t_inds) # still needed?
        util.update_image(pixels, xcoords, ycoords, color, t_inds)

        # save image and file sizes
        if record_pixels:
            pixels_vst[t_step] = pixels
        if remove_inactive: # here, pixels are removed when they become inactive
            act_coor_inds = cpart.active_coord_inds(times[t_step-1], times[t_step])
            num_active_pix[t_step] = len(act_coor_inds)
            pixels_out = cpart.white_image(2)
            coords = cpart.coords_offset()[:, act_coor_inds]
            pixels_out[coords[1], coords[0]] = pixels[coords[1], coords[0]]
        else:
            num_active_pix[t_step] = cpart.coords.shape[1]
            pixels_out = pixels

        num_differing_pixels[t_step] = count_image_differences(pixels, pixels_previous, cpart, act_coor_inds) if (num_active_pix[t_step] != 0 or cpart.is_rectangle) else 0
        pixels_previous = np.copy(pixels)

        namecore = 'canvaspart_time{:06d}'.format(int(times[t_step]))
        _, impath_png, impath_bmp = util.pixels_to_image(pixels_out, out_path_time, namecore + '.png', namecore + '.bmp')
        print(impath_png)

        file_size_png[t_step] = util.get_file_size(impath_png)
        file_size_bmp[t_step] = util.get_file_size(impath_bmp)
        if delete_bmp:
            os.remove(impath_bmp)
        if delete_png:
            os.remove(impath_png)

        if show_plot:
            if t_step>-1:
                if len(ax.shape) == 2:
                    ax_single = ax[rowcount, colcount]
                else:
                    ax_single = ax[t_step]
                ax_single.axis('off')
                plot.show_canvas_part(util.get_rgb(pixels), ax=ax_single)

                if colcount < 9:
                    colcount += 1
                else:
                    colcount = 0
                    rowcount += 1

    if print_progress:
        print('          produced', num_time_steps, 'images vs time ')
    return [file_size_bmp, file_size_png, pixels_vst, num_active_pix, num_differing_pixels, t_inds_list # do we still need t_inds_list?
    ]
