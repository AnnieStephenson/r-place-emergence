import numpy as np
import zlib
import math
from hilbertcurve.hilbertcurve import HilbertCurve
from numba import jit
import sweetsourcod
from sweetsourcod.lempel_ziv import lempel_ziv_complexity
from sweetsourcod.hilbert import get_hilbert_mask
from sweetsourcod.zipper_compress import get_comp_size_bytes
from sweetsourcod.block_entropy import block_entropy
from Levenshtein import distance as levenshtein_distance
import scipy
from rplacem import var as var
import os
import pickle
import collections
import skimage


def calc_size(pixels):
    '''
    Calculates the size of the uncompressed pixels
    '''
    data = pixels.tobytes()
    len_data = len(data)
    return len_data


def calc_compressed_size(pixels, flattening='hilbert_sweetsourcod', compression='LZ77'):
    '''
    Calculates the compressed file size of an image

    Parameters
    ---------
    pixels: numpy array
        2d pixel info
    flattening: string, optional
        Type of flattening used. if "hilbert_sweetsourcod", uses the hilbert curve flattening
        from the sweetsourcod package developed by Stefan Martiniani. if "ravel", then flattens
        into a simple 1d array without taking advantage of the original 2d structure of the image.
        "hilbert" is another implementation of the hilbert curve method.

    Returns
    -------
    len_compressed: float
        Length of the compressed image array.

    '''
    if flattening[0:7] == 'hilbert':
        # for any hilbert method, have to set up pixel padding and define exponent of 2
        power2 = math.ceil(math.log2(max(pixels.shape[0], pixels.shape[1])))
        pixels_pad = np.zeros((2**power2, 2**power2))
        pixels_pad[:pixels.shape[0], :pixels.shape[1]] = pixels

    if flattening == 'hilbert_sweetsourcod':
        lattice_boxv = np.asarray(pixels_pad.shape[::-1])
        hilbert_mask = get_hilbert_mask(lattice_boxv)
        pixels_flat = mask_array(pixels_pad.ravel(), hilbert_mask).astype('uint8')

    if flattening == 'hilbert_pkg':
        pixels_flat = np.zeros(len(pixels_pad.flatten()))
        hc = HilbertCurve(power2, 2)
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                h_index = hc.distance_from_point([i, j])
                pixels_flat[h_index] = pixels[i, j]

    elif flattening == 'ravel':
        pixels_flat = pixels.flatten()

    if compression == 'LZ77':
        len_compressed, _ = lempel_ziv_complexity(pixels_flat, 'lz77')

    if compression == 'LZ78':
        len_compressed = lempel_ziv_complexity(pixels_flat, 'lz78')

    if compression == 'DEFLATE':
        data = pixels_flat.tobytes()
        compressed_data = zlib.compress(data)
        len_compressed = len(compressed_data)

    return len_compressed

def normalize_entropy(entropy, area, time, 
                      compression="LZ77", flattening="ravel", num_iter=10, len_max=1000, minmax_entropy=None):
    '''
    Normalize the entropy
    '''
    (f_entropy_min, 
    f_entropy_max_8,
    f_entropy_max_16, 
    f_entropy_max_24, 
    f_entropy_max_32) = calc_min_max_entropy(compression=compression, 
                                             flattening=flattening,
                                             num_iter=num_iter,
                                             len_max=len_max)
    if time < var.TIME_16COLORS:
        entropy_norm = (entropy - f_entropy_min(area)) / (f_entropy_max_8(area) - f_entropy_min(area))
    if time >= var.TIME_16COLORS and time < var.TIME_24COLORS:
        entropy_norm = (entropy - f_entropy_min(area)) / (f_entropy_max_16(area) - f_entropy_min(area))
    elif time >= var.TIME_24COLORS and time < var.TIME_32COLORS:
        entropy_norm = (entropy - f_entropy_min(area)) / (f_entropy_max_24(area) - f_entropy_min(area))
    elif time >= var.TIME_32COLORS:
        entropy_norm = (entropy - f_entropy_min(area)) / (f_entropy_max_32(area) - f_entropy_min(area))

    return entropy_norm

def calc_min_max_entropy(compression="LZ77",
                         flattening="ravel",
                         num_iter=10,
                         len_max=1000):
    """
    Calculate the min and max entropy for 16, 24, and 32 colors
    if the functions are already saved in a pickle file, load them
    """
    files = os.listdir(var.DATA_PATH)
    if 'entropy_min_max.pkl' in files:
        with open(os.path.join(var.DATA_PATH, 'entropy_min_max.pkl'), 'rb') as f:
            f_entropy_min, f_entropy_max_8, f_entropy_max_16, f_entropy_max_24, f_entropy_max_32 = pickle.load(f)
    
    else:
        # define array sequences for each length, using 16, 32, or all black
        squares = np.arange(1, len_max)**2
        entropy_8 = np.zeros(len(squares))
        entropy_16 = np.zeros(len(squares))
        entropy_24 = np.zeros(len(squares))
        entropy_32 = np.zeros(len(squares))

        for i in range(0, len(squares)):
            sq_len = squares[i]

            flat_black = np.zeros(sq_len)
            flat_black = flat_black.reshape(int(np.sqrt(sq_len)),
                                            int(np.sqrt(sq_len)))
            entropy_8_range = np.zeros(num_iter)
            entropy_16_range = np.zeros(num_iter)
            entropy_24_range = np.zeros(num_iter)
            entropy_32_range = np.zeros(num_iter)
            # entropy_black_range = np.zeros(num_iter)

            for j in range(num_iter):
                # entropy 8
                rand_flat_8 = np.random.choice(8, size=sq_len)
                rand_flat_8= rand_flat_8.reshape(int(np.sqrt(sq_len)),
                                                    int(np.sqrt(sq_len)))
                len_comp = calc_compressed_size(rand_flat_8,
                                                    flattening=flattening,
                                                    compression=compression)
                entropy_8_range[j] = len_comp / sq_len

                # entropy 16
                rand_flat_16 = np.random.choice(16, size=sq_len)
                rand_flat_16 = rand_flat_16.reshape(int(np.sqrt(sq_len)),
                                                    int(np.sqrt(sq_len)))
                len_comp = calc_compressed_size(rand_flat_16,
                                                    flattening=flattening,
                                                    compression=compression)
                entropy_16_range[j] = len_comp / sq_len

                # entropy 24
                rand_flat_24 = np.random.choice(24, size=sq_len)
                rand_flat_24 = rand_flat_24.reshape(int(np.sqrt(sq_len)),
                                                    int(np.sqrt(sq_len)))
                len_comp = calc_compressed_size(rand_flat_24,
                                                    flattening=flattening,
                                                    compression=compression)
                entropy_24_range[j] = len_comp / sq_len

                # entropy 32
                rand_flat_32 = np.random.choice(32, size=sq_len)
                rand_flat_32 = rand_flat_32.reshape(int(np.sqrt(sq_len)),
                                                    int(np.sqrt(sq_len)))
                len_comp = calc_compressed_size(rand_flat_32,
                                                    flattening=flattening,
                                                    compression=compression)
                entropy_32_range[j] = len_comp / sq_len

                # len_comp = ent.calc_compressed_size(flat_black, flattening=flattening, compression=compression)
                # entropy_black_range[j] = len_comp/sq_len
            entropy_8[i] = np.mean(entropy_8_range)
            entropy_16[i] = np.mean(entropy_16_range)
            entropy_24[i] = np.mean(entropy_24_range)
            entropy_32[i] = np.mean(entropy_32_range)

        f_entropy_min = scipy.interpolate.interp1d(squares,
                                                1 / squares,
                                                kind="linear")
        f_entropy_max_8 = scipy.interpolate.interp1d(squares,
                                                    entropy_8,
                                                    kind="linear")
        f_entropy_max_16 = scipy.interpolate.interp1d(squares,
                                                    entropy_16,
                                                    kind="linear")
        f_entropy_max_24 = scipy.interpolate.interp1d(squares,
                                                    entropy_24,
                                                    kind="linear")
        f_entropy_max_32 = scipy.interpolate.interp1d(squares,
                                                    entropy_32,
                                                    kind="linear")
        # entropy_black[i] = np.mean(entropy_black_range)
        # flat_white = np.ones(sq_len, dtype='int32')
        # flat_white = flat_white.reshape(int(np.sqrt(sq_len)), int(np.sqrt(sq_len)))
        # len_comp = ent.calc_compressed_size(flat_white, flattening=flattening, compression=compression)
        # entropy_white[i] = len_comp/sq_lenm

        # save the functions to a pickle file
        with open(os.path.join(var.DATA_PATH, 'entropy_min_max.pkl'), 'wb') as f:
            pickle.dump((f_entropy_min, f_entropy_max_8, f_entropy_max_16, f_entropy_max_24, f_entropy_max_32), f)

    return f_entropy_min, f_entropy_max_8, f_entropy_max_16, f_entropy_max_24, f_entropy_max_32


def randomize(pixels):
    '''
    Randomizes the order pixels in an image
    '''
    # Get the size of the array
    size = pixels.size

    # Create a shuffled index array
    shuffled_idx = np.random.permutation(size)
    shuffled_pix = np.reshape(pixels.flat[shuffled_idx], pixels.shape)

    return shuffled_pix


def decimate(pixels, delta, dec_start=0):
    '''
    Decimate the image at interval delta
    '''
    if len(pixels.shape) == 1:
        pixel_dec = pixels[dec_start::delta]
    if len(pixels.shape) == 2:
        pixel_dec = pixels[dec_start::delta, dec_start::delta]
    return pixel_dec


def calc_Q(pixels, delta, flattening='hilbert_sweetsourcod', compression='LZ77'):
    '''
    calculate the Q value (scales with compressibility) for a given interval delta.
    '''
    kappa = np.zeros(delta)
    kappa_shuff = np.zeros(delta)
    for i in range(delta):  # loop over all possible decimation start points
        # Get the decimated pixels
        pixel_dec = decimate(pixels, delta, dec_start=i)

        # Calculate the compressed size of the decimated image
        kappa[i] = calc_compressed_size(pixel_dec, flattening=flattening, compression=compression)

        # Calcualte the compressed size of the randomly shuffled image
        shuffled_pix = randomize(pixel_dec)
        kappa_shuff[i] = calc_compressed_size(shuffled_pix, flattening=flattening, compression=compression)

    # Calculate the Q value
    Q = 1 - np.mean(kappa)/np.mean(kappa_shuff)
    return Q


def calc_Q_delta_vst(canvas_part_stat, flattening='hilbert_sweetsourcod', compression='LZ77'):
    '''
    Calculate the Q vs delta curve for each instantaneous image in the canvas_part_stat
    '''
    # Get the image pixels over time.
    pixels_vst = canvas_part_stat.true_image

    # Get the range of deltas.
    delta = np.arange(1, math.floor(pixels_vst.shape[1]/2)+1)

    # Calculate Q
    n_tlims = canvas_part_stat.n_t_bins + 1
    q = np.zeros((n_tlims, len(delta)))
    for j in range(n_tlims):  # loop through the times
        for i in range(len(delta)):  # loop through the deltas
            q[j, i] = calc_Q(pixels_vst[j], delta[i], flattening=flattening, compression=compression)

    return q, delta


def compute_complexity_multiscale(image_vst, scales=None):
    """
    Compute normalized Zhang complexity K over time from a 3D stack of 2D images using mode-based
    coarse-graining with fast bincount mode computation.

    Parameters:
        image_vst (np.ndarray): 3D array of shape (T, H, W), where each [t] is a 2D image with non-negative integers.
        scales (list[int] or None): Optional list of block sizes. If None, uses powers of 2 up to min(H, W) // 4.

    Returns:
        np.ndarray of shape (T,) with normalized complexity values for each time step.
    """
    T, H, W = image_vst.shape
    K_norm_over_time = np.zeros(T)

    # Auto-generate scales if not provided
    if scales is None:
        max_scale = max(1, min(H, W) // 4)
        powers = int(np.log2(max_scale)) + 1
        scales = [1] + [2 ** i for i in range(1, powers + 1) if 2 ** i <= max_scale]

    num_scales = len(scales)

    for t in range(T):
        image = image_vst[t]
        entropies = np.zeros(num_scales)
        complexity = 0

        for i, r in enumerate(scales):
            # Crop image to nearest multiple of r
            h_crop = (image.shape[0] // r) * r
            w_crop = (image.shape[1] // r) * r
            cropped = image[:h_crop, :w_crop]

            # Extract non-overlapping blocks
            blocks = skimage.util.view_as_blocks(cropped, block_shape=(r, r))
            h, w = blocks.shape[:2]
            flattened_blocks = blocks.reshape(h, w, -1)

            # mode via bincount
            def fast_mode(x):
                return np.argmax(np.bincount(x))

            modes = np.apply_along_axis(fast_mode, -1, flattened_blocks)

            # Compute histogram of mode values
            counts = collections.Counter(modes.flatten())
            total = sum(counts.values())
            probs = np.array([c / total for c in counts.values()])

            # Shannon entropy
            S_r = -np.sum(probs * np.log(probs))
            entropies[i] = S_r

            # Complexity K = sum of r^2 * S(r)
            complexity += (r ** 2) * S_r

        # Normalize by number of pixels
        N = image.shape[0] * image.shape[1]
        K_norm_over_time[t] = complexity / N

    return K_norm_over_time


def compute_complexity_levenshtein(image_vst):
    """
    Computes the Levenshtein-based spatial complexity CL for each time slice of a 3D image.

    Parameters:
        image_vst (3D numpy array): array of shape (T, R, C) with values 0–31 representing colors.

    Returns:
        1D numpy array of length T with the CL value at each time step
    """
    T, R, C = image_vst.shape
    CL_t = np.zeros(T, dtype=np.float32)

    for t in range(T):
        image = image_vst[t]
        V_r = np.zeros(R - 1, dtype=np.uint8)
        V_c = np.zeros(C - 1, dtype=np.uint8)

        # Row-wise Levenshtein distances
        for i in range(R - 1):
            row1 = ''.join(map(chr, image[i]))
            row2 = ''.join(map(chr, image[i + 1]))
            V_r[i] = levenshtein_distance(row1, row2)

        # Column-wise Levenshtein distances
        for j in range(C - 1):
            col1 = ''.join(map(chr, image[:, j]))
            col2 = ''.join(map(chr, image[:, j + 1]))
            V_c[j] = levenshtein_distance(col1, col2)

        # Outer product and mean for CL
        A = np.outer(V_r, V_c)
        CL_t[t] = A.mean()

    return CL_t



###############################################################################################
# The following functions come from:
# https://github.com/martiniani-lab/sweetsourcod/blob/master/examples/simple.py
# which is an example file from the sweetsourcod package developed by Stefano Martiniani et al.
###############################################################################################

@jit(nopython=True)
def mask_array(lattice, mask):
    return np.array([lattice[i] for i in mask])


def _get_entropy_rate(c, nsites, norm=1, alphabetsize=2, method='lz77'):
    """
    :param c: number of longest previous factors (lz77) or unique words (lz78)
    :param norm: normalization constant, usually the filesize per character of a random binary sequence of the same length
    :param method: lz77 or lz78
    :return: entropy rate h
    """
    if method == 'lz77':
        h = (c * np.log2(c) + 2 * c * np.log2(nsites / c)) / nsites
        h /= norm
    elif method == 'lz78':
        h = c * (np.log2(alphabetsize) + np.log2(c)) / nsites
        h /= norm
    else:
        raise NotImplementedError
    return h

def get_entropy_rate_lz77(x, extrapolate=True):
    # now with LZ77 we compute the number of longest previous factors c, and the entropy rate h
    # note that lz77 also outputs the sum of the logs of the factors which is approximately
    # equal to the compressed size of the file
    nsites = len(x)
    if extrapolate:
        random_binary_sequence = np.random.randint(0, 2, nsites, dtype='uint8')
        c_bin, sumlog_bin = lempel_ziv_complexity(random_binary_sequence, 'lz77')
        h_bound_bin = _get_entropy_rate(c_bin, nsites, norm=1, alphabetsize=np.unique(x).size, method='lz77')
        h_sumlog_bin = sumlog_bin
    else:
        h_bin = 1
        h_sumlog_bin = nsites
    c, h_sumlog = lempel_ziv_complexity(x, 'lz77')
    h_bound = _get_entropy_rate(c, nsites, norm=h_bound_bin, alphabetsize=np.unique(x).size, method='lz77')
    h_sumlog /= h_sumlog_bin
    return h_bound, h_sumlog


def get_entropy_rate_lz78(x, extrapolate=True):
    nsites = len(x)
    if extrapolate:
        random_binary_sequence = np.random.randint(0, 2, nsites, dtype='uint8')
        c_bin = lempel_ziv_complexity(random_binary_sequence, 'lz78')
        h_bound_bin = _get_entropy_rate(c_bin, nsites, norm=1, alphabetsize=np.unique(x).size, method='lz78')
    else:
        h_bin = 1
    c = lempel_ziv_complexity(x, 'lz78')
    h_bound = _get_entropy_rate(c, nsites, norm=h_bound_bin, alphabetsize=np.unique(x).size, method='lz78')
    return h_bound


def get_entropy_rate_bbox(x, extrapolate=True, algorithm='deflate', **kwargs):
    nsites = len(x)
    if extrapolate:
        random_binary_sequence = np.random.randint(0, 2, nsites, dtype='uint8')
        enc, _, _ = get_comp_size_bytes(random_binary_sequence, algorithm=algorithm, **kwargs)
        h_bin = 8 * enc / nsites
    else:
        h_bin = 1
    enc, _, _ = get_comp_size_bytes(x, algorithm=algorithm, **kwargs)
    h = 8 * enc / nsites
    return h / h_bin
