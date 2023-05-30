import numpy as np
import pandas as pd


def max_box_size(im_width, im_height):
    '''
    Calculates the maximum box size for fractal dimension algorithm.
    The smallest maximum box size is 3 pixels, to enforce at least 3 points
    to fit the line to (Since in that case, the boxes sizes will be 1, 2,
    and 3 pixels).
    '''

    # maximum box size is a quarter diameter
    max_box_size = round(0.1*min(im_width, im_height))

    # require at least 3 points to fit line
    if max_box_size < 3:
        max_box_size = 3
    return max_box_size


def get_box_inds(box_size, im_height, im_width, shift_avg=True):
    '''
    parameters
    ----------
    box_size: numpy array,
    im_height: float
        number of pixels in y-dimension
    im_width: float
        number of pixels in x-dimension
    shift_avg: boolean, optional
        If True, calcalutes four sets of box_inds so that box edges are aligned with each
        corner.

    returns
    -------
    box_inds_res: numpy array or list of numpy arrays
        The box indices for each pixel. If shift_avg is True, box_ind_res is a list of
        four numpy arrays with the indices aligned with each corner of the image/composition

    '''
    num_box_per_col = int(np.ceil(im_width / box_size))
    num_box_per_row = int(np.ceil(im_height / box_size))
    num_box_tot = num_box_per_col * num_box_per_row
    grid_width = num_box_per_col * box_size
    grid_height = num_box_per_row * box_size

    box_cols = np.arange(1, num_box_per_col + 1)
    box_inds_row = np.repeat(box_cols, box_size)
    box_inds = np.tile(box_inds_row, grid_height)
    box_inds = np.reshape(box_inds, (grid_height, len(box_inds_row)))
    box_col0 = np.arange(1, num_box_tot + 1, num_box_per_col)
    box_shift = np.repeat(box_col0, box_size)
    box_shift = np.repeat(box_shift[:, np.newaxis], grid_width, axis=1)
    box_inds = box_inds + box_shift

    cutoff_h = grid_height - im_height
    cutoff_w = grid_width - im_width
    if shift_avg:
        box_inds_crop1 = box_inds[0:im_height, 0:im_width]
        box_inds_crop2 = box_inds[cutoff_h:im_height + cutoff_h, cutoff_w:im_width+cutoff_w]
        box_inds_crop3 = box_inds[cutoff_h:im_height + cutoff_h, 0:im_width]
        box_inds_crop4 = box_inds[0:im_height, cutoff_w:im_width+cutoff_w]
        box_inds_res = [box_inds_crop1, box_inds_crop2, box_inds_crop3, box_inds_crop4]
    else:
        box_inds_res = box_inds[0:im_height, 0:im_width]

    return box_inds_res


def count_num_box_touch(true_image, shift_avg=True):
    '''
    Counts the  number of boxes touched by pixels of each color at each time step
    and at each box size. The box sizes are calculated within the function based
    on the size of the image.
    '''
    n_t_lims = true_image.shape[0]
    im_height = true_image.shape[1]
    im_width = true_image.shape[2]

    max_size = max_box_size(im_width, im_height)
    box_size = np.arange(1, max_size + 1)
    if shift_avg:
        num_boxes_touched1 = np.zeros((len(box_size), n_t_lims, 32))
        num_boxes_touched2 = np.zeros((len(box_size), n_t_lims, 32))
        num_boxes_touched3 = np.zeros((len(box_size), n_t_lims, 32))
        num_boxes_touched4 = np.zeros((len(box_size), n_t_lims, 32))
    else:
        num_boxes_touched = np.zeros((len(box_size), n_t_lims, 32), dtype=int)

    for k in range(len(box_size)):
        box_inds = get_box_inds(box_size[k], im_height, im_width, shift_avg=shift_avg)
        if shift_avg:
            box_inds1, box_inds2, box_inds3, box_inds4 = box_inds
        for i in range(n_t_lims):
            for j in range(32):
                color_bool = true_image[i] == j
                if shift_avg:
                    num_boxes_touched1[k, i, j] = num_box_touch_from_color_bool(color_bool, box_inds1)
                    num_boxes_touched2[k, i, j] = num_box_touch_from_color_bool(color_bool, box_inds2)
                    num_boxes_touched3[k, i, j] = num_box_touch_from_color_bool(color_bool, box_inds3)
                    num_boxes_touched4[k, i, j] = num_box_touch_from_color_bool(color_bool, box_inds4)
                else:
                    num_boxes_touched[k, i, j] = num_box_touch_from_color_bool(color_bool, box_inds)

    if shift_avg:
        num_boxes_touched = (num_boxes_touched1 + num_boxes_touched2 + num_boxes_touched3 + num_boxes_touched4)/4

    return num_boxes_touched, box_size


def num_box_touch_from_color_bool(color_bool, box_inds):
    '''
    Calculates the number of boxes touched
    from the boolean array containing where the pixels of a given color are located
    and the array of box indices accross all pixels
    '''
    color_boxes = color_bool*box_inds
    color_boxes = color_boxes[color_boxes != 0]
    return len(np.unique(color_boxes))


def calc_fractal_dim(box_size, num_boxes_touched):
    '''
    Calculates the fractal dimension by fitting a line to the log of the box size and the log of the number of boxes touched.
    The fractal dimension is the slope of the linear fit.
    '''
    n_box_shp = num_boxes_touched.shape
    num_boxes_rshp = num_boxes_touched.reshape((len(box_size), n_box_shp[1]*n_box_shp[2]))
    log_num_box = np.log(num_boxes_rshp)
    log_num_box[np.where(log_num_box == -np.inf)] = 0
    coeffs = np.polyfit(np.log(box_size), log_num_box, 1)
    fractal_dim = -coeffs[0]
    fractal_dim = fractal_dim.reshape((n_box_shp[1], n_box_shp[2]))
    return fractal_dim


def get_dom_colors(true_image):
    '''
    Get the dominant colors at each time step for later use in calculating
    fractal dimension averages weighted by color dominance.

    Returns
    --------
    dom_color_frac: numpy array
        fraction of pixels of each color at each time step, ordered by descending color
    dom_colors: numpy array
        colors of pixels ordered by dominance at each time step
    color_frac: numpy array
        fraction of pixels of each color at each time step

    '''
    n_t_lims = true_image.shape[0]

    color_frac = np.zeros((n_t_lims, 32))
    dom_color_frac = np.zeros((n_t_lims, 32))
    num_pix = true_image[0].shape[0]*true_image[0].shape[1]
    for j in range(n_t_lims):
        for i in range(32):
            dim1, dim2 = np.where(true_image[j] == i)
            color_frac[j, i] = len(dim1)/num_pix
    dom_colors = np.argsort(-color_frac, axis=1)
    dom_color_frac = color_frac[np.arange(color_frac.shape[0]), dom_colors[:, 0]]

    return dom_color_frac, dom_colors, color_frac


def calc_fractal_dim_values(dom_colors, color_frac, fractal_dim):
    '''
    Calculates fractal dimension summary variables
    '''
    fractal_dim_mean = np.mean(fractal_dim, axis=1)
    fractal_dim_mask = np.ma.masked_equal(fractal_dim, 0)
    fractal_dim_mask_mean = np.ma.mean(fractal_dim_mask, axis=1)
    fractal_dim_mask_median = np.ma.median(fractal_dim_mask, axis=1)
    fractal_dim_dom = fractal_dim[np.arange(fractal_dim.shape[0]), dom_colors[:, 0]]
    fractal_dim_weighted = np.sum(fractal_dim*color_frac, axis=1)

    return [fractal_dim_mean,
            fractal_dim_mask,
            fractal_dim_mask_mean,
            fractal_dim_mask_median,
            fractal_dim_dom,
            fractal_dim_weighted]


def calc_from_image(true_image, shift_avg=True):
    '''
    Calculates and returns selected fractal dimension summary variables
    given only the input image over time
    '''
    # get the number of boxes touched per box size
    num_boxes_touched, box_size = count_num_box_touch(true_image, shift_avg=shift_avg)

    # calculate the fractal dimension for each color
    fractal_dim = calc_fractal_dim(box_size, num_boxes_touched)

    # calculate dominant colors for fractal dimension summary values
    dom_color_frac, dom_colors, color_frac = get_dom_colors(true_image)

    # calcualte fractal dimension summary values
    [fractal_dim_mean,
     fractal_dim_mask,
     fractal_dim_mask_mean,
     fractal_dim_mask_median,
     fractal_dim_dom,
     fractal_dim_weighted] = calc_fractal_dim_values(dom_colors, color_frac, fractal_dim)

    return fractal_dim_mask_mean, fractal_dim_mask_median, fractal_dim_dom, fractal_dim_weighted
