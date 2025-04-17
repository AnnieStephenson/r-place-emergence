import cProfile
import os
import sys
import numpy as np
import glob
from rplacem import var as var
import json
import shutil
from PIL import Image
import pickle


def get_all_pixel_changes(data_file=var.FULL_DATA_FILE,
                          times_before_whiteout=True):
    '''
    load all the pixel change data and put it in a numpy array for easy access

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
                                                 ('moderator', np.bool_)]))
    pixel_changes_all['seconds'] = np.array(pixel_changes_all_npz['seconds'])
    pixel_changes_all['xcoor'] = np.array(pixel_changes_all_npz['pixelXpos'])
    pixel_changes_all['ycoor'] = np.array(pixel_changes_all_npz['pixelYpos'])
    pixel_changes_all['user'] = np.array(pixel_changes_all_npz['userIndex'])
    pixel_changes_all['color'] = np.array(pixel_changes_all_npz['colorIndex'])
    pixel_changes_all['moderator'] = np.array(
        pixel_changes_all_npz['moderatorEvent'])

    if times_before_whiteout:
        pixel_changes_all = pixel_changes_all[pixel_changes_all['seconds'] < var.TIME_WHITEOUT]
    return pixel_changes_all


def get_rgb(col_idx):
    ''' Returns the (r,g,b) triplet corresponding to the input color index '''
    return var.COLIDX_TO_RGB[col_idx]


def get_file_size(path):
    ''' Gets the length of a file in bytes'''
    f = open(path, "rb").read()
    byte_array = bytearray(f)
    return len(byte_array)


def check_time(statement, sort='cumtime'):
    '''
    parameters:
    -----------
    statement: string

    '''
    cProfile.run(statement, sort=sort)


def equalize_list_sublengths(l):
    ''' Fill out each sublist-member of the list up to the length of the longest member, with redundant values (so that it can be a np.array)'''
    maxlen = max(len(v) for v in l)
    for i in range(0,len(l)):
        l[i] += [l[i][0]] * max(maxlen - len(l[i]), 0)
    return l


def pixels_to_image(pix, dir='', save_name='', save_name2=''):
    ''' Transform the 2d array of color indices into an image object, and saves it if (save_name!='') '''
    im = Image.fromarray( get_rgb(pix).astype(np.uint8) )

    if save_name != '':
        try:
            os.makedirs( os.path.join(var.FIGS_PATH, dir) )
        except OSError:
            pass
        impath1 = os.path.join(var.FIGS_PATH, dir, save_name)
        impath2 = os.path.join(var.FIGS_PATH, dir, save_name2)
        im.save(impath1)
        if save_name2 != '':
            im.save(impath2)

    return im, impath1, impath2


def save_movie(image_path,
               fps=1,
               movie_tool='moviepy',
               codec='libx264',
               video_type='mp4',
               logger=None):
    '''
    Save movie of .png images in the path.

    parameters
    ----------
    image_path: string
        Path where .png images can be found
    movie_tool: string
        Movie handling package you want to use
        values can be 'moviepy', 'ffmpeg-python', or 'ffmpeg'
        You must have the corresponding packages installed for this to work.
        'moviepy' and 'ffmpeg-python' refer to python packages. 'ffmpeg' refers
        to software that must be installed on your system without requiring a
        specific python package.
    '''
    image_files = list(np.sort(glob.glob(os.path.join(image_path, '*.png'))))
    print(image_files[:3])
    print(f"# of image files: {len(image_files)}")
    png_name0 = os.path.basename(image_files[0][0:15])
    movie_name = png_name0 + '_fps' + str(fps)
    movie_file = os.path.join(image_path, movie_name) + '.' + video_type

    if movie_tool == 'moviepy':
        #if 'imsc' not in sys.modules:
        #    import moviepy.video.io.ImageSequenceClip as imsc
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip(image_files, fps=fps)
        #clip = imsc.ImageSequenceClip(image_files, fps=fps)
        clip.duration = len(image_files) / float(fps) 
        print("clip.fps =", clip.fps)
        clip.write_videofile(movie_file, codec=codec, logger=logger, fps=float(fps))

    if movie_tool == 'ffmpeg-python':
        # frames may not be in order
        if movie_tool not in sys.modules:
            import ffmpeg
        (ffmpeg.input(image_path + '/*.png', pattern_type='glob', framerate=fps)
         .output(movie_file, vcodec=codec).overwrite_output().run())

    if movie_tool == 'ffmpeg':
        # frames may not be in order
        os.system('ffmpeg -framerate ' + str(fps) + '-pattern_type glob -i *.png' + codec + ' -y ' + movie_file)


def update_image(image, xcoords, ycoords, color, t_inds=None):
    '''Update the pixels of the [image] using the 1d arrays [xcoords], [ycoords], [color].
    These latter arrays are taken at indices [t_inds].
    For each (x,y) pixel, one needs to keep only the last pixel change. '''
    if len(t_inds) == 0:
        return
    if t_inds is None:
        t_inds = np.arange(0, len(color))

    color_inv = (color[t_inds])[::-1]
    ycoords_inv = (ycoords[t_inds])[::-1]
    xcoords_inv = (xcoords[t_inds])[::-1]
    xycoords_inv = np.column_stack((xcoords_inv, ycoords_inv))
    xyidx_unique, idx_first = np.unique(xycoords_inv, return_index=True, axis=0) # keeping the index of the first occurence in the reverse-order array
    image[xyidx_unique[:,1] , xyidx_unique[:,0]] = color_inv[idx_first]

def load_atlas():
    '''
    Load the composition atlas and return the atlas and the number of entries in the atlas
    '''
    atlas_path = os.path.join(var.DATA_PATH, 'atlas.json')
    atlas_file = open(atlas_path)
    atlas = json.load(atlas_file)

    atlas_size = len(list(atlas))
    return atlas, atlas_size

def make_dir(path, renew=False):
    '''
    Makes directory only if it does not exist yet.
    If [renew], removes the directory contents if it exists.
    Returns a bool saying if the dir existed beforehand
    '''
    try:
        os.makedirs(path)
        res = False
    except OSError:  # empty directory if it already exists
        if renew:
            shutil.rmtree(path)
        res = True

    return res

def merge_pickles(file_list, file_out):
    out = []
    for file in file_list:
        with open(file, 'rb') as f:
            out += pickle.load(f)

    with open(os.path.join(var.DATA_PATH, file_out), 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

def divide_treatzero(a, b, escape=1, escape_posnum=10):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = a / b
    res[np.where((b == 0) & (a == 0))] = escape
    res[np.where((b == 0) & (a != 0))] = escape_posnum

    return res

def pixel_integrated_stats(a, fallback, percentile10=False):
    '''
    Calculates the mean, median, 90th percentile and mean of highest decile for input array.
    Considers 10th percentile and lowest decile if [percentile10]
    Returns [fallback] for empty array
    '''
    n = len(a)
    if n == 0:
        return np.array([fallback, fallback, fallback, fallback])
    else:
        sorted = np.sort(a)
        return np.array([np.mean(a),
                         sorted[int(n/2)],
                         sorted[(1 if percentile10 else -1) * int(n/10)],
                         np.mean(sorted[ :int(n/10) ] if percentile10 else sorted[ -int(n/10) :])])
