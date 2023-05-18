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
        power2 = math.ceil(math.log2(pixels.shape[0]))
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
