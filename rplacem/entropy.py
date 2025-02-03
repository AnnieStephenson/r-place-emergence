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
import scipy
from rplacem import var as var
import os
import pickle


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
        with open(os.path.join(var.DATA_PATH, '..', 'entropy_min_max.pkl'), 'rb') as f:
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
