import cProfile
import os
import sys
import numpy as np
import glob
import rplacem.variables_rplace2022 as var

def get_all_pixel_changes(data_file=var.FULL_DATA_FILE):
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
                                                 ('xcoor', np.uint16), 
                                                 ('ycoor', np.uint16), 
                                                 ('user', np.uint32), 
                                                 ('color', np.uint8), 
                                                 ('moderator', np.bool_)])
                                 )
    pixel_changes_all['seconds'] = np.array(pixel_changes_all_npz['seconds'])
    pixel_changes_all['xcoor'] = np.array(pixel_changes_all_npz['pixelXpos'])
    pixel_changes_all['ycoor'] = np.array(pixel_changes_all_npz['pixelYpos'])
    pixel_changes_all['user'] = np.array(pixel_changes_all_npz['userIndex'])
    pixel_changes_all['color'] = np.array(pixel_changes_all_npz['colorIndex'])
    pixel_changes_all['moderator'] = np.array(pixel_changes_all_npz['moderatorEvent'])

    return pixel_changes_all

def get_file_size(path):
    ''' Gets the length of a file in bytes'''
    f = open(path, "rb").read()
    byte_array = bytearray(f)
    return len(byte_array)

def check_time(statement, sort = 'cumtime'):
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

def save_movie(image_path,
               fps=1,
               movie_tool='moviepy',
               codec='libx264',
               video_type='mp4'):
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
    png_name0 = os.path.basename(image_files[0][0:-15])
    movie_name = png_name0 + '_fps' + str(fps)
    movie_file = os.path.join(image_path, movie_name) + '.' + video_type

    if movie_tool == 'moviepy':
        if 'imsc' not in sys.modules:
            import moviepy.video.io.ImageSequenceClip as imsc
        clip = imsc.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(movie_file,  codec=codec)

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
    if t_inds is None:
        t_inds = np.arange(0, len(color))
    
    color_inv = (color[t_inds])[::-1]
    ycoords_inv = (ycoords[t_inds])[::-1]
    xcoords_inv = (xcoords[t_inds])[::-1]
    xycoords_inv = np.column_stack((xcoords_inv, ycoords_inv))
    xyidx_unique, idx_first = np.unique(xycoords_inv, return_index=True, axis=0) # keeping the index of the first occurence in the reverse-order array
    image[xyidx_unique[:,1] , xyidx_unique[:,0]] = color_inv[idx_first]