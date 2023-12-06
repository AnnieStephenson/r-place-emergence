import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rplacem.entropy as ent
import rplacem.canvas_part as cp
import pandas as pd
import pickle
import rplacem.utilities as util
import scipy
import rplacem.globalvariables_peryear as vars
var = vars.var


def plot_loglog_fit(x_data_unfilt, y_data_unfilt,
                    subsample=True,
                    x_data_filter_min=0,
                    y_data_filter_min=0,
                    x_data_filter_max=None,
                    y_data_filter_max=None,
                    data_color=[0.3, 0.3, 0.3],
                    line_color=[0, 0, 0],
                    markersize=2,
                    alpha_data=0.15,
                    alpha_line=0.7,
                    linewidth=2.5,
                    nbins=None,
                    max_bin_size=10,
                    subsample_count=100,
                    plot_sliding_window_ave=True,
                    x_bin=False,
                    x_line_max=None,
                    semilog=False):
    '''
    Plots the data on a loglog plot with linear fit
    '''

    x_data_unfilt = np.array(x_data_unfilt)
    y_data_unfilt = np.array(y_data_unfilt)

    # filter
    x_data = x_data_unfilt[(x_data_unfilt > x_data_filter_min) & (y_data_unfilt > y_data_filter_min)]
    y_data = y_data_unfilt[(x_data_unfilt > x_data_filter_min) & (y_data_unfilt > y_data_filter_min)]

    if y_data_filter_max is not None:
        x_data = x_data[y_data < y_data_filter_max]
        y_data = y_data[y_data < y_data_filter_max]
    if x_data_filter_max is not None:
        y_data = y_data[x_data < x_data_filter_max]
        x_data = x_data[x_data < x_data_filter_max]

    if nbins is None:
        nbins = round(len(x_data)/100)
        print(nbins)
    if semilog is True:
        plt.semilogy(x_data, y_data, '.', color=data_color, alpha=alpha_data, markersize=markersize)
    if semilog == 'x':
        plt.semilogx(x_data, y_data, '.', color=data_color, alpha=alpha_data, markersize=markersize)
    else:
        plt.loglog(x_data, y_data, '.', color=data_color, alpha=alpha_data, markersize=markersize)
    sns.despine()

    if subsample:
        slopes = []
        intercepts = []
        for i in range(subsample_count):
            if not x_bin:
                x_data_sub, y_data_sub = subsample_data_ybins(x_data, y_data, nbins=nbins, max_bin_size=max_bin_size)
            if x_bin:
                y_data_sub, x_data_sub = subsample_data_ybins(y_data, x_data, nbins=nbins, max_bin_size=max_bin_size)
            log_x_data = np.log10(x_data_sub)
            log_y_data = np.log10(y_data_sub)
            coefficients = np.polyfit(log_x_data, log_y_data, 1)
            s, i = coefficients
            slopes.append(s)
            intercepts.append(i)
        slope = np.mean(slopes)
        intercept = np.mean(intercepts)

    else:
        log_x_data = np.log10(x_data)
        log_y_data = np.log10(y_data)
        coefficients = np.polyfit(log_x_data, log_y_data, 1)
        slope, intercept = coefficients
    print('intercept: ' + str(intercept))
    print('exponent: ' + str(slope))

    if x_line_max is None:
        x_line_max = np.max(x_data)

    if x_data_filter_min != 0:
        x_line_min = np.log10(x_data_filter_min)
    else:
        x_line_min = 0

    x_line_data = np.logspace(x_line_min, np.log10(x_line_max))
    plt.plot(x_line_data, 10 ** (intercept) * x_line_data ** (slope),
             color=line_color,
             alpha=alpha_line,
             linewidth=linewidth)
    return x_data, y_data


def subsample_data_ybins(x_data, y_data,
                         nbins=100,
                         max_bin_size=10):
    '''
    Subsamples data in different bins to account for uneven data distribution in x or y
    '''

    log_x_data = np.log10(x_data)
    log_y_data = np.log10(y_data)

    hist_values, bin_edges = np.histogram(log_y_data, bins=nbins)

    # Get the values within each bin
    y_values_clip = np.array([])
    x_values_clip = np.array([])
    for i in range(len(bin_edges) - 1):
        indices = np.where((log_y_data >= bin_edges[i]) & (log_y_data < bin_edges[i + 1]))[0]
        if len(indices) < max_bin_size:
            max_samp = len(indices)
        else:
            max_samp = max_bin_size
        sampled_inds = np.random.choice(indices, size=max_samp)
        x_values = x_data[sampled_inds]
        y_values = y_data[sampled_inds]
        x_values_clip = np.concatenate((x_values_clip, x_values))
        y_values_clip = np.concatenate((y_values_clip, y_values))
    return x_values_clip, y_values_clip


def calc_min_max_entropy(compression='LZ77',
                         flattening='ravel',
                         num_iter=10,
                         len_max=370
                         ):
    '''
    Calculate the min and max entropy for 16 and 32 colors
    '''
    # define array sequences for each length, using 16, 32, or all black
    squares = np.arange(1, len_max)**2
    entropy_16 = np.zeros(len(squares))
    entropy_32 = np.zeros(len(squares))

    for i in range(0, len(squares)):
        sq_len = squares[i]

        flat_black = np.zeros(sq_len)
        flat_black = flat_black.reshape(int(np.sqrt(sq_len)), int(np.sqrt(sq_len)))

        entropy_16_range = np.zeros(num_iter)
        entropy_32_range = np.zeros(num_iter)
        # entropy_black_range = np.zeros(num_iter)

        for j in range(num_iter):
            rand_flat_16 = np.random.choice(16, size=sq_len)
            rand_flat_16 = rand_flat_16.reshape(int(np.sqrt(sq_len)), int(np.sqrt(sq_len)))
            len_comp = ent.calc_compressed_size(rand_flat_16, flattening=flattening, compression=compression)
            entropy_16_range[j] = len_comp/sq_len

            rand_flat_32 = np.random.choice(32, size=sq_len)
            rand_flat_32 = rand_flat_32.reshape(int(np.sqrt(sq_len)), int(np.sqrt(sq_len)))
            len_comp = ent.calc_compressed_size(rand_flat_32, flattening=flattening, compression=compression)
            entropy_32_range[j] = len_comp/sq_len

            # len_comp = ent.calc_compressed_size(flat_black, flattening=flattening, compression=compression)
            # entropy_black_range[j] = len_comp/sq_len

        entropy_16[i] = np.mean(entropy_16_range)
        entropy_32[i] = np.mean(entropy_32_range)


    f_entropy_min = scipy.interpolate.interp1d(squares, 1/squares, kind='linear')
    f_entropy_max_16 = scipy.interpolate.interp1d(squares, entropy_16, kind='linear')
    f_entropy_max_32 = scipy.interpolate.interp1d(squares, entropy_32, kind='linear')
        # entropy_black[i] = np.mean(entropy_black_range)
        # flat_white = np.ones(sq_len, dtype='int32')
        # flat_white = flat_white.reshape(int(np.sqrt(sq_len)), int(np.sqrt(sq_len)))
        # len_comp = ent.calc_compressed_size(flat_white, flattening=flattening, compression=compression)
        # entropy_white[i] = len_comp/sq_len
    return f_entropy_min, f_entropy_max_16, f_entropy_max_32

def calc_user_attn(canvas_comp):
    pixch_user_sorted = canvas_comp.pixel_changes[canvas_comp.pixch_sortuser]
    pixch_user_sorted = pixch_user_sorted[pixch_user_sorted['active'] == 1]
    users, user_inds = np.unique(pixch_user_sorted['user'][::-1], return_index=True)
    user_end_inds = len(pixch_user_sorted) - user_inds
    tmin, tmax = canvas_comp.min_max_time()
    max_num_px_ch = (var.TIME_WHITEOUT-tmin)/300.
    attn_frac = np.zeros(len(users))
    npix = canvas_comp.num_pix()
    for j in range(0, len(users)):
        if j == 0:
            coord_inds = pixch_user_sorted['coord_index'][0: user_end_inds[j]]
            times = pixch_user_sorted['seconds'][0: user_end_inds[j]]
        else:
            coord_inds = pixch_user_sorted['coord_index'][user_end_inds[j-1]: user_end_inds[j]]
            times = pixch_user_sorted['seconds'][user_end_inds[j-1]: user_end_inds[j]]
        num_pxch = len(times)
        attn_frac[j] = num_pxch/max_num_px_ch
    mean_attn = np.mean(attn_frac)
    med_attn = np.median(attn_frac)
    pix_norm_attn = np.mean(np.sort(attn_frac)[0:int(npix)]) # mean of the n_pix most attentive players

    return mean_attn, med_attn, pix_norm_attn

# define function for user distance
def calc_user_distance(canvas_comp):
    pixch_user_sorted = canvas_comp.pixel_changes[canvas_comp.pixch_sortuser]
    pixch_user_sorted = pixch_user_sorted[pixch_user_sorted['active'] == 1]
    users, user_inds = np.unique(pixch_user_sorted['user'][::-1], return_index=True)
    user_end_inds = len(pixch_user_sorted) - user_inds

    dist_total = np.zeros(len(users))
    dist_mean = np.zeros(len(users))
    dist_central_mean = np.zeros(len(users))
    delta_t = np.zeros(len(users))
    mean_timediff = np.zeros(len(users))
    speed_ave = np.zeros(len(users))
    for j in range(0, len(users)):
        if j==0:
            coord_inds = pixch_user_sorted['coord_index'][0: user_end_inds[j]]
            times = pixch_user_sorted['seconds'][0: user_end_inds[j]]
        else:
            coord_inds = pixch_user_sorted['coord_index'][user_end_inds[j-1]: user_end_inds[j]]
            times = pixch_user_sorted['seconds'][user_end_inds[j-1]: user_end_inds[j]]
        delta_t[j] = times[-1] - times[0]
        mean_timediff[j] = np.mean(np.diff(times))
        coords_x, coords_y = canvas_comp.coords[:, coord_inds]
        x_mean = np.mean(coords_x)
        y_mean = np.mean(coords_y)
        diffx = np.abs(np.diff(coords_x))
        diffy = np.abs(np.diff(coords_y))
        dist_subseq = np.sqrt(diffx**2 + diffy**2)
        dist_central = np.sqrt((x_mean-coords_x)**2 + (y_mean-coords_y)**2)
        dist_central_mean[j] = np.mean(dist_central)
        if len(dist_subseq) == 0:
            dist_total[j] = np.nan
            dist_mean[j] = np.nan
        else:
            dist_total[j] = np.sum(dist_subseq)
            dist_mean[j] = np.mean(dist_subseq)
        speed_ave[j] = dist_total[j]/delta_t[j]

    return dist_total, dist_mean, dist_central_mean, delta_t, mean_timediff, speed_ave

def calc_recovery_time(cpart_stat, frac, ref_frac):
    void_start_inds = np.where(frac-ref_frac > 0.7)[0]
    if len(void_start_inds) > 1:
        void_start_ind = void_start_inds[0]
    else:
        void_start_ind = 0

    void_end_ind = np.where(frac-ref_frac < 0.1)[0]
    potential_stop_inds = np.where(void_end_ind > void_start_ind)[0]
    if len(potential_stop_inds) > 1 and len(np.atleast_1d(void_start_inds)) > 1:
        void_end_ind = void_end_ind[potential_stop_inds[0]]
    else:
        void_start_ind = 0
        void_end_ind = 0
    start_void = cpart_stat.t_lims[void_start_ind]
    end_void = cpart_stat.t_lims[void_end_ind]

    recovery_time = end_void - start_void
    return recovery_time



def get_comp_scaling_data(canvas_parts_file='canvas_parts.pkl',
                          canvas_parts_stats_file='canvas_part_stats_sw.pkl',
                          filename='reddit_place_composition_list_extended_sw.csv'):

    '''
    Calculate and save the scaling data as a .csv file

    Notes:
      There are two main types of results we've looked at here:

      One is based on a canvas_part_stats that includes only one time interval,
      which is the entire time the composition was active on the canvas,
      and the analysis is done on the most stable image.

      The other is based on a canvas_part_stats that looks at time steps in reference to a sliding window.
      This makes it difficult to give a single value for many of the quantities of the canvas.
      For example, for stability, there is a new value at each interval.
      For entropy of the reference image, there is also a new values at each interval, since the sliding window is always changing.
      Therefore, many of the saved values will be arrays in this case, and plotting the scaling of something like the entropy
      will mean we have to decide exactly what to plot. Perhaps, in some cases, we'd wish to plot the mean? or the max? or the min? or maybe all of these.
      This motivates saving the entire array for the timestep, which also suggests that we may want to replace some of the summed values with the entire array instead.
    '''

    with open(canvas_parts_file, 'rb') as file:
        canvas_comp_list = pickle.load(file)

    with open(canvas_parts_stats_file, 'rb') as file:
        canvas_part_stats_list = pickle.load(file)

    atlas, atlas_size = util.load_atlas()

    names = []
    subreddit = []
    size_pixels = []
    n_users_total = []
    n_defense_changes = []
    n_attack_changes = []
    n_defenseonly_users = []
    n_attackonly_users = []
    n_bothattdef_users = []
    instab = []
    streamer = []
    alliance = []
    tmin = []
    tmax = []
    tmin_quad = []
    stability = []
    compressed_size = []
    entropy = []
    fractal_dim = []
    dist_totals = []
    dist_means = []
    dist_central_means = []
    delta_ts = []
    speed_aves = []
    recovery_black = []
    recovery_purple = []
    mean_attns = []
    med_attns = []
    pix_norm_attns = []

    # other metrics of interest: instability, entropy,
    num_iter = len(canvas_part_stats_list)
    print(num_iter)
    for i in range(num_iter):
        print(i)
        cpart_stat = canvas_part_stats_list[i]
        canvas_comp = canvas_comp_list[i]

        att_ch_thresh = 0#np.sum(cpart_stat.n_changes.val - cpart_stat.n_defense_changes.val)
        def_ch_thresh = np.sum(cpart_stat.n_defense_changes.val)

        if len(canvas_comp.info.links) != 0 and 'subreddit' in canvas_comp.info.links:
            subreddit.append(canvas_comp.info.links['subreddit'])
        else:
            subreddit.append('NA')

        streamer_flag = 0
        if ('streamer' in canvas_comp.info.description) or ('stream' in canvas_comp.info.description):
            streamer_flag = 1

        alliance_flag = 0
        if ('alliance' in canvas_comp.info.description) or ('ally' in canvas_comp.info.description) or ('allies' in canvas_comp.info.description):
            alliance_flag = 1
        elif subreddit[-1] != 'NA' and (('Alliance' in canvas_comp.info.links) or ('ally' in canvas_comp.info.links)):
            alliance_flag = 1

        #dist_total, dist_mean, dist_central_mean, delta_t, speed_ave = calc_user_distance(canvas_comp)
        mean_attn, med_attn, pix_norm_attn = calc_user_attn(canvas_comp)

        recovery_time_black = calc_recovery_time(cpart_stat,
                                                 cpart_stat.frac_black_px.val,
                                                 cpart_stat.frac_black_ref.val)

        recovery_time_purple = calc_recovery_time(cpart_stat,
                                                  cpart_stat.frac_purple_px.val, cpart_stat.frac_purple_ref.val)

        # add values to composition lists
        recovery_purple.append(recovery_time_purple)
        recovery_black.append(recovery_time_black)
        names.append(canvas_comp.info.atlasname)
        streamer.append(streamer_flag)
        alliance.append(alliance_flag)
        n_users_total.append(np.sum(cpart_stat.n_users_total))
        size_pixels.append(canvas_comp.num_pix())
        n_defense_changes.append(def_ch_thresh)
        n_attack_changes.append(att_ch_thresh)
        n_defenseonly_users.append(np.sum(cpart_stat.n_defenseonly_users.val))
        n_attackonly_users.append(np.sum(cpart_stat.n_attackonly_users.val))
        n_bothattdef_users.append(np.sum(cpart_stat.n_bothattdef_users.val))
        tmin.append(cpart_stat.tmin)
        tmax.append(cpart_stat.tmax)
        tmin_quad.append(canvas_comp.tmin_quadrant())
        stability.append(cpart_stat.stability.val)  # may be an array
        instab.append(np.mean(cpart_stat.instability_norm.val))  # may be an array
        compressed_size.append(cpart_stat.size_compr_stab_im.val)  # may be an array
        entropy.append(cpart_stat.entropy_stab_im.val)  # may be an array
        fractal_dim.append(cpart_stat.fractal_dim_weighted.val)  # may be an array
        mean_attns.append(mean_attn)
        med_attns.append(med_attn)
        pix_norm_attns.append(pix_norm_attn)
        #dist_totals.append(dist_total)
        #dist_means.append(dist_mean)
        #dist_central_means.append(dist_central_mean)
        #delta_ts.append(delta_t)
        #speed_aves.append(speed_ave)

    data = {
            'Name': names,
            'Subreddit': subreddit,
            'Size (pixels)': size_pixels,
            'Num users total': n_users_total,
            'Num defense-only users': n_defenseonly_users,
            'Num attack-only users': n_attackonly_users,
            'Num attack-defense users': n_bothattdef_users,
            'Num defense changes': n_defense_changes,
            'Num attack changes': n_attack_changes,
            'Instability': instab,
            'Streamer': streamer,
            'Alliance': alliance,
            'Start time (s)': tmin,
            'End time (s)': tmax,
            'Start time quadrant (s)': tmin_quad,
            'Stability': stability,
            'Compressed size': compressed_size,
            'Entropy': entropy,
            'Recovery time (black)': recovery_black,
            'Recovery time (purple)': recovery_purple,
            'Mean Attention': mean_attns,
            'Median Attention': med_attns,
            'Pixel Norm Attention': pix_norm_attns
            }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    return df


def load_comp_scaling_data(filename='reddit_place_composition_list_extended.csv'):
    '''
    load the dataframe of scaling data and split between different community types
    '''
    df = pd.read_csv(filename)

    df_s = df[df['Streamer'] == 1]
    df_a = df[df['Alliance'] == 1]
    df_o = df[(df['Alliance'] == 0) & (df['Streamer'] == 0)]

    return df, df_s, df_a, df_o
