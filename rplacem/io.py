from operator import mod
from re import T
import cv2
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
import PIL as pil
import seaborn as sns

datapath = os.path.join(os.getcwd(),'data/')
fname_base = '2022_place_canvas_history-0000000000'

class Artwork:
    '''
    attributes
    ----------
    border_path: 
    pixel_changes: 2d pandas array
        columns: timestamp, user_id, x_coord, y_coord, R, G, B
    
    methods
    -------
    get_border():
    get_coords():
    draw_border():
    get_change_coords():
    draw_over_time():
    '''
    def __init__(self, border_path, pixel_changes):
        self.border_path = border_path
        self.pixel_changes = pixel_changes
    
def hex_to_rgb(hex_str):
    '''
    Turns hex color string to rgb
    
    parameters
    ----------
    
    returns
    -------
    
    '''
    hex_str = hex_str.lstrip('#')
    len_hex = len(hex_str)
    return tuple(int(hex_str[i:i + len_hex // 3], 16) for i in range(0, len_hex, len_hex // 3))

def get_path(id, data_path=os.path.join(os.getcwd(),'data')): 
    '''
    Read info from json file
    
    parameters
    ----------
    id: float or int

    returns
    -------
    path: list of points that describes the countour of the selected artwork
    '''
    atlas_path = os.path.join(data_path,'atlas.json')
    artwork_classification_file = open(atlas_path)
    atlas = json.load(artwork_classification_file)
    
    for i in range(len(atlas)):
        if atlas[i]['id']==id:
            id_index = i
            
    print('artwork name: ' + str(atlas[id_index]['name']))
    print('path: ' + str(atlas[id_index]['path']))
    paths = atlas[id_index]['path']
    path0 = np.array(list(paths.values()))[0] # first
    return paths, path0

def draw_path(path0, show_path =False):
    '''
    plot paths
    
    parameters
    ----------
    path: list of points describing contour of selected artwork
    
    returns
    --------
    mask
    '''
    img = np.ones((1000,1000))
    mask = np.zeros((1000,1000))
    cv2.fillPoly(mask, pts=[path0], color = [1,1,1])
    masked_img = cv2.bitwise_and(img, mask)

    if show_path==True:
        plt.figure()
        plt.imshow(masked_img)

    y_art_coords, x_art_coords = np.where(masked_img==1)
    return x_art_coords, y_art_coords


def get_all_pixel_change_coords(data_file, data_path=os.path.join(os.getcwd(),'data')):
    '''
    plots the pixel values for a given artwork path, at the ending time of the csv
    
    parameters
    ----------
    
    returns
    -------
    pixel_image: image of the pixel values
    '''
    print('data file: ' + data_file + '\n')
    data_file_path = os.path.join(data_path, data_file)
    tiles = pd.read_csv(data_file_path)
    
    # get coordinate array that includes all the coordinates of changed pixels from the .csv file
    # colors array includes all the hex color values of changed pixels from the .csv file
    color_changes_hex = np.array(tiles['pixel_color']) 
    coords_change_string_array = np.array(tiles['coordinate'])
    timestamp = np.array(tiles['timestamp'])
    user_id = np.array(tiles['user_id']) 
    coords_change_string_array=coords_change_string_array.astype(str)

    # deal with the admin-created rectangle coordinates    
    coords_change_split_xy = np.chararray.split(coords_change_string_array,sep=',')
    coords_change_split_xy_array = np.array([np.array(i) for i in coords_change_split_xy], dtype=object)
    coords_change_2d_array = np.zeros((coords_change_split_xy_array.shape[0],2))
    
    for i in range(0,  len(coords_change_split_xy_array)):
        if coords_change_split_xy_array[i].size>2:
            rect_corner_coords = coords_change_split_xy_array[i].astype(int)
            rect_range_x = np.arange(rect_corner_coords[0], rect_corner_coords[2]+1)
            rect_range_y = np.arange(rect_corner_coords[1], rect_corner_coords[3]+1)
            rect_coords_x, rect_coords_y = np.meshgrid(rect_range_x, rect_range_y)
            rect_coords_x = rect_coords_x.flatten()
            rect_coords_y = rect_coords_y.flatten()
            rect_coords = np.transpose(np.array([rect_coords_x, rect_coords_y]))
            coords_change_2d_array = np.insert(coords_change_2d_array, i, rect_coords, axis=0)
            
            mod_rect_color_hex = np.repeat('#FFFFFF', rect_coords.shape[0])
            color_changes_hex = np.insert(color_changes_hex, i, mod_rect_color_hex)

            mod_user_id = np.repeat(user_id[i], rect_coords.shape[0])
            user_id = np.insert(user_id, i, mod_user_id)

            mod_timestamp = np.repeat(timestamp[i], rect_coords.shape[0])
            timestamp = np.insert(timestamp, i, mod_timestamp)
        else:
            coords_change_2d_array[i] = coords_change_split_xy_array[i].astype(int)
            
    # split up into x and y for user 
    x_change_coords = coords_change_2d_array[:,0]
    y_change_coords = coords_change_2d_array[:,1]

    # put these into a pandas df: sorted timestamp, x_coord, y_coord, user_id, color_hex
    # sort the df by datetime
    # return the df
    timestamp = pd.to_datetime(list(timestamp), infer_datetime_format=True)
    file_pixel_changes = pd.DataFrame(data={'timestamp': timestamp, 
                                            'x_coord': x_change_coords, 
                                            'y_coord': y_change_coords,
                                            'user_id': user_id, 
                                            'color_hex': color_changes_hex})
    
    
    return file_pixel_changes #x_change_coords, y_change_coords, color_changes_hex
    
    
def get_artwork_colors(x_art_coords, y_art_coords, file_pixel_changes):     
    '''
    
    parameters
    ----------
    
    returns
    --------
    
    
    '''
    x_change_coords = np.array(file_pixel_changes['x_coord'])
    y_change_coords = np.array(file_pixel_changes['y_coord'])
    color_changes_hex = np.array(file_pixel_changes['color_hex'])
    timestamp = np.array(file_pixel_changes['timestamp'])
    user_id = np.array(file_pixel_changes['timestamp'])

    art_change_index = np.where(np.isin(x_change_coords, x_art_coords) & np.isin(y_change_coords, y_art_coords))[0]
    
    colors_art_change_hex = color_changes_hex[art_change_index]
    y_art_change_coords = y_change_coords[art_change_index].astype(int) 
    x_art_change_coords = x_change_coords[art_change_index].astype(int)
    timestamp_art_change = timestamp[art_change_index]
    timestamp_art_change = pd.to_datetime(list(timestamp_art_change), infer_datetime_format=True)
    user_id_art_change = user_id[art_change_index]


    colors_art_change_r = np.zeros(len(colors_art_change_hex))
    colors_art_change_g = np.zeros(len(colors_art_change_hex))
    colors_art_change_b = np.zeros(len(colors_art_change_hex))

    for i in range(len(colors_art_change_hex)):
        (colors_art_change_r[i],
         colors_art_change_g[i],
         colors_art_change_b[i]) = hex_to_rgb(colors_art_change_hex[i])
        
    colors_art_change = np.array([colors_art_change_r, colors_art_change_g, colors_art_change_b])

    # now make a new dataframe with the 
    artwork_pixel_changes = pd.DataFrame(data={'timestamp': timestamp_art_change, 
                                            'x_coord': x_art_change_coords, 
                                            'y_coord': y_art_change_coords,
                                            'user_id': user_id_art_change, 
                                            'color_R': colors_art_change_r,
                                            'color_G': colors_art_change_g,
                                            'color_B': colors_art_change_b})

    return artwork_pixel_changes #colors_art_change, x_art_change_coords, y_art_change_coords
        
        
def show_artwork(x_art_change_coords, y_art_change_coords, 
                 colors_art_change_r,
                 colors_art_change_g,
                 colors_art_change_b,
                 ax=None):  
    '''
    parameters
    ----------
    
    returns
    -------
    '''
    #colors_art_change_r, colors_art_change_g, colors_art_change_b = colors_art_change_list
    img = np.zeros((1000,1000))
    img_r = np.ones((1000,1000))
    img_g = np.ones((1000,1000))
    img_b = np.ones((1000,1000))
    img_r[x_art_change_coords.astype(int), y_art_change_coords.astype(int)] = colors_art_change_r/255
    img_g[x_art_change_coords.astype(int), y_art_change_coords.astype(int)] = colors_art_change_g/255
    img_b[x_art_change_coords.astype(int), y_art_change_coords.astype(int)] = colors_art_change_b/255
    img = np.swapaxes(np.array([img_r, img_g, img_b]), 0, 2)


    if ax==None:
        plt.figure(origin='upper')
        plt.imshow(img,origin='upper')
    else:
        ax.imshow(img, origin='upper')

def show_art_over_time(id_name, file_numbers, time_interval, 
                        total_time=82.5 # total time of rplace ~82.5 hrs need to check
                        ):
    '''
    
    parameters
    ----------
    time_interval: in hours
    
    returns
    -------
    '''
    artwork_pixel_changes_combined  = get_art_pixel_changes_over_time(id_name, file_numbers)
    timestamp = artwork_pixel_changes_combined['timestamp']
    timestamp = pd.to_datetime(list(timestamp), infer_datetime_format=True)
    time_delta = timestamp-timestamp[0]
    time_delta_hrs = time_delta.seconds/3600
   
    num_time_steps = int(np.ceil(total_time/time_interval))

    ncols = np.min([num_time_steps, 10])
    nrows = np.max([1, int(math.ceil(num_time_steps/10))])
    #fig = plt.figure(figsize=(10,nrows)) # height corresponds in inches to number of rows. auto dpi is 100
    #gs = fig.add_gridspec(nrows, ncols, hspace=0.05, wspace=0.05)
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    print(ax.shape)
    #ax = gs.subplots(sharex=True, sharey=True)
    rowcount = 0
    colcount = 0
    time_inds_list = []
    for i in range(1, nrows*ncols + 1):
        if len(ax.shape)==2:
            ax_single=ax[rowcount,colcount]
        else:
            ax_single = ax[i-1]
        ax_single.axis('off')
        if i < (num_time_steps + 1):
            time_inds = np.where(time_delta_hrs<=i*time_interval)[0]
            time_inds_list.append(time_inds)
            artwork_pixel_changes_time_integrated = artwork_pixel_changes_combined.iloc[time_inds,:]
            x_art_change_coords_integrated =  np.array(artwork_pixel_changes_time_integrated['x_coord'])
            y_art_change_coords_integrated =  np.array(artwork_pixel_changes_time_integrated['y_coord'])
            color_art_changes_integrated_r =  np.array(artwork_pixel_changes_time_integrated['color_R'])
            color_art_changes_integrated_g =  np.array(artwork_pixel_changes_time_integrated['color_G'])
            color_art_changes_integrated_b =  np.array(artwork_pixel_changes_time_integrated['color_B'])

            show_artwork(x_art_change_coords_integrated,
                        y_art_change_coords_integrated, 
                        color_art_changes_integrated_r,
                        color_art_changes_integrated_g,
                        color_art_changes_integrated_b, ax = ax_single)
            
        if colcount<9:
            colcount+=1
        else:
            colcount=0
            rowcount+=1
        path0, path = get_path(id_name)
        path_coords = np.array(max(path0.values()))
        x_min = np.min(path_coords[:,0])
        x_max = np.max(path_coords[:,0])
        y_min = np.min(path_coords[:,1])
        y_max = np.max(path_coords[:,1])
        plt.xlim([x_min, x_max])
        plt.ylim([y_max, y_min])

    return (artwork_pixel_changes_combined,
            time_inds_list)
    
def get_art_change_coords_colors(id_name, data_file):
    '''
    parameters
    ----------
    
    returns
    -------
    
    '''
    
    paths, path0 = get_path(id_name)

    # get the coordinates of all tiles within the artwork
    x_art_coords, y_art_coords = draw_path(path0)
    # get the coordinates of tiles that were changed in the dataset:
    file_pixel_changes = get_all_pixel_change_coords(data_file)
    #x_change_coords, y_change_coords, color_changes_hex = get_all_pixel_change_coords(data_file)

    # get the colors the tiles were changed to: color_changes
    # and the coordinates of the art tiles that were changed in the dataset: x_art_change_coords, y_art_change_coords
    artwork_pixel_changes = get_artwork_colors(x_art_coords, y_art_coords, file_pixel_changes)
                                                                    
    return artwork_pixel_changes
    
def get_art_pixel_changes_over_time(id_name, file_numbers):
    '''
    
    parameters
    ----------
    
    returns
    -------
    
    '''
    artwork_pixel_changes_combined = pd.DataFrame(data={'timestamp':[], 
                                                       'x_coord':[],
                                                       'y_coord': [],
                                                       'user_id':[],
                                                       'color_R':[],
                                                       'color_G':[],
                                                       'color_B':[]})

    for i in range(0, file_numbers.size):
        if file_numbers[i]<=9:
            extra_str = '0'
        else:
            extra_str = ''
        data_file = fname_base + extra_str + str(file_numbers[i]) + '.csv'
        artwork_pixel_changes = get_art_change_coords_colors(id_name, data_file)
        artwork_pixel_changes_combined = pd.concat([artwork_pixel_changes_combined, artwork_pixel_changes])

    # sort by datetime 
    artwork_pixel_changes_combined['timestamp']=pd.to_datetime(artwork_pixel_changes_combined['timestamp'], infer_datetime_format=True)
    artwork_pixel_changes_combined = artwork_pixel_changes_combined.sort_values(by=['timestamp']) 
        
    return artwork_pixel_changes_combined
    
    
def save_and_compress_artwork(id_name, artwork_pixel_changes_combined, time_inds_list, bmp=True, png=True):
    path0, path = get_path(id_name)
    path_coords = np.array(max(path0.values()))
    x_min = np.min(path_coords[:,0])
    x_max = np.max(path_coords[:,0])
    y_min = np.min(path_coords[:,1])
    y_max = np.max(path_coords[:,1])
    
    file_size_bmp = np.zeros(len(time_inds_list))
    file_size_png = np.zeros(len(time_inds_list))
    for i in range(0,len(time_inds_list)):
        im = pil.Image.new("RGB",(x_max-x_min + 1, y_max-y_min + 1),"white")
        artwork_pixel_changes_time_integrated = artwork_pixel_changes_combined.iloc[time_inds_list[i],:]
        x_art_change_coords_integrated =  np.array(artwork_pixel_changes_time_integrated['x_coord'])
        y_art_change_coords_integrated =  np.array(artwork_pixel_changes_time_integrated['y_coord'])
        color_art_changes_integrated_r =  np.array(artwork_pixel_changes_time_integrated['color_R'])
        color_art_changes_integrated_g =  np.array(artwork_pixel_changes_time_integrated['color_G'])
        color_art_changes_integrated_b =  np.array(artwork_pixel_changes_time_integrated['color_B'])

        pixels = im.load()
        colors = np.vstack((color_art_changes_integrated_r, 
                            color_art_changes_integrated_g, 
                            color_art_changes_integrated_b))
        for j in range(0, len(x_art_change_coords_integrated)):
            pixels[int(x_art_change_coords_integrated[j]-x_min), 
                   int(y_art_change_coords_integrated[j]-y_min)] = tuple(colors.transpose()[j].astype(int))
        
        if bmp:
            im.save('frames' + str(i) + '.bmp')
            with open('frames' + str(i) + '.bmp', "rb") as image_file:
                f = image_file.read()
                byte_array = bytearray(f)
            file_size_bmp[i] = len(byte_array)
            
        if png:
            im.save('compressed_frames' + str(i) + '.png', optimize = True)
            with open('compressed_frames' + str(i) + '.png', "rb") as image_file:
                f = image_file.read()
                byte_array = bytearray(f)
            file_size_png[i] = len(byte_array)
                
    return file_size_bmp, file_size_png
    
def plot_compression(file_size_bmp, file_size_png, time_interval, total_time):
    time = np.arange(time_interval, total_time + time_interval, time_interval)

    plt.figure()
    plt.plot(time, file_size_bmp, label='bmp')
    plt.plot(time, file_size_png, '--', label='png')
    sns.despine()
    plt.legend(frameon=False)
    plt.ylabel('file size (bytes)')
    plt.xlabel('time (hrs)')
    
    plt.figure()
    plt.plot(time, file_size_png/file_size_bmp)
    sns.despine()
    plt.ylabel('file size ratio')
    plt.xlabel('time (hrs)')
