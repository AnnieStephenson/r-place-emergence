import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import scipy.odr 

import rplacem.canvas_part as cp
import rplacem.entropy as ent
import rplacem.globalvariables_peryear as vars
import rplacem.utilities as util

var = vars.var

def linear_model(params, x):
    return params[0] * x + params[1]

def plot_loglog_fit(x_data_unfilt,
                    y_data_unfilt,
                    z_data_unfilt=None,
                    x_data_filter_min=0,
                    y_data_filter_min=0,
                    z_data_filter_min=0,
                    x_data_filter_max=None,
                    y_data_filter_max=None,
                    data_color=[0.3, 0.3, 0.3],
                    line_color=[0, 0, 0],
                    markersize=2,
                    alpha_data=0.15,
                    alpha_line=0.7,
                    alpha_error=0.3,
                    linewidth=2.5,
                    nbins=None,
                    bin_type='subsample', # 'average'
                    bin_axis='y', # 'x'
                    fit_type='OLS', # 'TLS'
                    max_bin_size=10,
                    n_subsample_iter=100,
                    plot_sliding_window_ave=True,
                    x_line_max=None,
                    semilog=False, fit=True):
    """
    Plots the data on a loglog plot with linear fit
    """

    x_data_unfilt = np.array(x_data_unfilt)
    y_data_unfilt = np.array(y_data_unfilt)
    if z_data_unfilt is not None:
        z_data_unfilt = np.array(z_data_unfilt)

    # filter
    x_data = x_data_unfilt[(x_data_unfilt > x_data_filter_min) & (
        y_data_unfilt > y_data_filter_min)]
    y_data = y_data_unfilt[(x_data_unfilt > x_data_filter_min) & (
        y_data_unfilt > y_data_filter_min)]
    if z_data_unfilt is not None:
        # [(z_data_unfilt > z_data_filter_min) & (z_data_unfilt > z_data_filter_min)]
        z_data = z_data_unfilt

    if y_data_filter_max is not None:
        x_data = x_data[y_data < y_data_filter_max]
        y_data = y_data[y_data < y_data_filter_max]
        if z_data_unfilt is not None:
            z_data = z_data[y_data < y_data_filter_max]
    if x_data_filter_max is not None:
        y_data = y_data[x_data < x_data_filter_max]
        x_data = x_data[x_data < x_data_filter_max]
        if z_data_unfilt is not None:
            z_data = z_data[x_data < x_data_filter_max]

    if nbins is None:
        nbins = round(len(x_data) / 100)
        print('number of ' + bin_axis + ' bins in which to subsample: ' + str(nbins))
    if semilog is True:
        plt.semilogy(
            x_data,
            y_data,
            ".",
            color=data_color,
            alpha=alpha_data,
            markersize=markersize,
        )
    if semilog == "x":
        plt.semilogx(
            x_data,
            y_data,
            ".",
            color=data_color,
            alpha=alpha_data,
            markersize=markersize,
        )
    if z_data_unfilt is not None:
        plt.scatter(
            x_data,
            y_data,
            c=np.log10(z_data),
            cmap="viridis",
            s=markersize,
            alpha=alpha_data,
        )
        plt.gca().set_yscale("log")
        plt.gca().set_xscale("log")
    else:
        plt.loglog(
            x_data,
            y_data,
            ".",
            color=data_color,
            alpha=alpha_data,
            markersize=markersize,
        )
    sns.despine()

    if bin_type!='subsample':
        n_subsample_iter=1

    slopes = []
    intercepts = []
    for i in range(n_subsample_iter):
        if bin_axis=='y':
            if bin_type=='subsample':
                x_data_bin, y_data_bin = subsample_data_ybins(x_data, y_data, nbins=nbins, max_bin_size=max_bin_size)
            if bin_type=='average':
                x_data_bin, y_data_bin = average_data_ybins(x_data, y_data, nbins=nbins, max_bin_size=max_bin_size)
        if bin_axis=='x':
            if bin_type=='subsample':
                y_data_bin, x_data_bin = subsample_data_ybins(y_data, x_data, nbins=nbins, max_bin_size=max_bin_size)
            if bin_type=='average':
                y_data_bin, x_data_bin = average_data_ybins(y_data, x_data, nbins=nbins, max_bin_size=max_bin_size)
        log_x_data = np.log10(x_data_bin)
        log_y_data = np.log10(y_data_bin)
        #plt.figure()
        #plt.loglog(log_x_data, log_y_data, '.')
        if fit_type=='OLS':
            coefficients = np.polyfit(log_x_data, log_y_data, 1)
            s, inter = coefficients
        if fit_type=='TLS':
            data = scipy.odr.RealData(log_x_data, log_y_data)
            model = scipy.odr.Model(linear_model)
            odr = scipy.odr.ODR(data, model, beta0=[1.0, 0.0])  # Initial guess for parameters
            result = odr.run()
            params = result.beta
            s= params[0]
            inter = params[1]
        slopes.append(s)
        intercepts.append(inter)
    slope = np.mean(slopes)
    slope_std = np.std(slopes)
    intercept = np.mean(intercepts)
    intercept_std = np.std(intercepts)


    print("intercept: " + str(intercept) + ' +- ' + str(intercept_std))
    print("exponent: " + str(slope) + ' +- ' + str(slope_std))

    if x_line_max is None:
        x_line_max = np.max(x_data)

    if np.min(x_data) > 0:
        x_line_min = np.log10(np.min(x_data))
    else:
        x_line_min = 0.1

    x_line_data = np.logspace(x_line_min, np.log10(x_line_max))
    plt.plot(
        x_line_data,
        10**(intercept) * x_line_data**(slope),
        color=line_color,
        alpha=alpha_line,
        linewidth=linewidth,
    )
    if linewidth == 0:
        alpha_error = 0
    plt.fill_between(
        x_line_data, 
        10**(intercept - intercept_std) * x_line_data**(slope - slope_std), 
        10**(intercept + intercept_std) * x_line_data**(slope + slope_std) , 
        linewidth=0,
        color=line_color, alpha=alpha_error)

    if z_data_unfilt is not None:
        data = x_data, y_data, z_data
    else:
        data = x_data, y_data
    return data

def average_data_ybins(x_data, y_data, nbins=80, max_bin_size=10):
    log_x_data = np.log10(x_data)
    log_y_data = np.log10(y_data)

    hist_values, bin_edges = np.histogram(log_y_data, bins=nbins)
    hist_values, bin_edges = np.histogram(log_y_data, bins=nbins)
    y_values_clip = np.array([])
    x_values_clip = np.array([])
    for i in range(len(bin_edges) - 1):
        indices = np.where((log_y_data >= bin_edges[i])
                           & (log_y_data < bin_edges[i + 1]))[0]
        y_values = np.array([np.mean(y_data[indices])])
        x_values = np.array([np.mean(x_data[indices])])
        if not np.isnan(x_values[0]):
            x_values_clip = np.concatenate((x_values_clip, x_values))
            y_values_clip = np.concatenate((y_values_clip, y_values))
    return x_values_clip, y_values_clip

def subsample_data_ybins(x_data, y_data, nbins=100, max_bin_size=10):
    """
    Subsamples data in different bins to account for uneven data distribution in x or y
    """

    log_x_data = np.log10(x_data)
    log_y_data = np.log10(y_data)

    hist_values, bin_edges = np.histogram(log_y_data, bins=nbins)
    # Get the values within each bin
    y_values_clip = np.array([])
    x_values_clip = np.array([])
    for i in range(len(bin_edges) - 1):
        indices = np.where((log_y_data >= bin_edges[i])
                           & (log_y_data < bin_edges[i + 1]))[0]
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


def calc_min_max_entropy(compression="LZ77",
                         flattening="ravel",
                         num_iter=10,
                         len_max=370):
    """
    Calculate the min and max entropy for 16 and 32 colors
    """
    # define array sequences for each length, using 16, 32, or all black
    squares = np.arange(1, len_max)**2
    entropy_16 = np.zeros(len(squares))
    entropy_32 = np.zeros(len(squares))

    for i in range(0, len(squares)):
        sq_len = squares[i]

        flat_black = np.zeros(sq_len)
        flat_black = flat_black.reshape(int(np.sqrt(sq_len)),
                                        int(np.sqrt(sq_len)))

        entropy_16_range = np.zeros(num_iter)
        entropy_32_range = np.zeros(num_iter)
        # entropy_black_range = np.zeros(num_iter)

        for j in range(num_iter):
            rand_flat_16 = np.random.choice(16, size=sq_len)
            rand_flat_16 = rand_flat_16.reshape(int(np.sqrt(sq_len)),
                                                int(np.sqrt(sq_len)))
            len_comp = ent.calc_compressed_size(rand_flat_16,
                                                flattening=flattening,
                                                compression=compression)
            entropy_16_range[j] = len_comp / sq_len

            rand_flat_32 = np.random.choice(32, size=sq_len)
            rand_flat_32 = rand_flat_32.reshape(int(np.sqrt(sq_len)),
                                                int(np.sqrt(sq_len)))
            len_comp = ent.calc_compressed_size(rand_flat_32,
                                                flattening=flattening,
                                                compression=compression)
            entropy_32_range[j] = len_comp / sq_len

            # len_comp = ent.calc_compressed_size(flat_black, flattening=flattening, compression=compression)
            # entropy_black_range[j] = len_comp/sq_len

        entropy_16[i] = np.mean(entropy_16_range)
        entropy_32[i] = np.mean(entropy_32_range)

    f_entropy_min = scipy.interpolate.interp1d(squares,
                                               1 / squares,
                                               kind="linear")
    f_entropy_max_16 = scipy.interpolate.interp1d(squares,
                                                  entropy_16,
                                                  kind="linear")
    f_entropy_max_32 = scipy.interpolate.interp1d(squares,
                                                  entropy_32,
                                                  kind="linear")
    # entropy_black[i] = np.mean(entropy_black_range)
    # flat_white = np.ones(sq_len, dtype='int32')
    # flat_white = flat_white.reshape(int(np.sqrt(sq_len)), int(np.sqrt(sq_len)))
    # len_comp = ent.calc_compressed_size(flat_white, flattening=flattening, compression=compression)
    # entropy_white[i] = len_comp/sq_len
    return f_entropy_min, f_entropy_max_16, f_entropy_max_32


def calc_user_attn(canvas_comp):
    pixch_user_sorted = canvas_comp.pixel_changes[canvas_comp.pixch_sortuser]
    pixch_user_sorted = pixch_user_sorted[pixch_user_sorted["active"] == 1]
    users, user_inds = np.unique(pixch_user_sorted["user"][::-1],
                                 return_index=True)
    user_end_inds = len(pixch_user_sorted) - user_inds
    tmin, tmax = canvas_comp.min_max_time()
    max_num_px_ch = (var.TIME_WHITEOUT - tmin) / 300.0
    attn_frac = np.zeros(len(users))
    npix = canvas_comp.num_pix()
    for j in range(0, len(users)):
        if j == 0:
            coord_inds = pixch_user_sorted["coord_index"][0:user_end_inds[j]]
            times = pixch_user_sorted["seconds"][0:user_end_inds[j]]
        else:
            coord_inds = pixch_user_sorted["coord_index"][
                user_end_inds[j - 1]:user_end_inds[j]]
            times = pixch_user_sorted["seconds"][
                user_end_inds[j - 1]:user_end_inds[j]]
        num_pxch = len(times)
        attn_frac[j] = num_pxch / max_num_px_ch
    mean_attn = np.mean(attn_frac)
    med_attn = np.median(attn_frac)
    pix_norm_attn = np.mean(np.sort(
        attn_frac)[0:int(npix)])  # mean of the n_pix most attentive players

    return mean_attn, med_attn, pix_norm_attn


# define function for user distance
def calc_user_distance(canvas_comp):
    pixch_user_sorted = canvas_comp.pixel_changes[canvas_comp.pixch_sortuser]
    pixch_user_sorted = pixch_user_sorted[pixch_user_sorted["active"] == 1]
    users, user_inds = np.unique(pixch_user_sorted["user"][::-1],
                                 return_index=True)
    user_end_inds = len(pixch_user_sorted) - user_inds

    dist_total = np.zeros(len(users))
    dist_mean = np.zeros(len(users))
    dist_central_mean = np.zeros(len(users))
    delta_t = np.zeros(len(users))
    mean_timediff = np.zeros(len(users))
    speed_ave = np.zeros(len(users))
    for j in range(0, len(users)):
        if j == 0:
            coord_inds = pixch_user_sorted["coord_index"][0:user_end_inds[j]]
            times = pixch_user_sorted["seconds"][0:user_end_inds[j]]
        else:
            coord_inds = pixch_user_sorted["coord_index"][
                user_end_inds[j - 1]:user_end_inds[j]]
            times = pixch_user_sorted["seconds"][
                user_end_inds[j - 1]:user_end_inds[j]]
        delta_t[j] = times[-1] - times[0]
        mean_timediff[j] = np.mean(np.diff(times))
        coords_x, coords_y = canvas_comp.coords[:, coord_inds]
        x_mean = np.mean(coords_x)
        y_mean = np.mean(coords_y)
        diffx = np.abs(np.diff(coords_x))
        diffy = np.abs(np.diff(coords_y))
        dist_subseq = np.sqrt(diffx**2 + diffy**2)
        dist_central = np.sqrt((x_mean - coords_x)**2 + (y_mean - coords_y)**2)
        dist_central_mean[j] = np.mean(dist_central)
        if len(dist_subseq) == 0:
            dist_total[j] = np.nan
            dist_mean[j] = np.nan
        else:
            dist_total[j] = np.sum(dist_subseq)
            dist_mean[j] = np.mean(dist_subseq)
        speed_ave[j] = dist_total[j] / delta_t[j]

    return dist_total, dist_mean, dist_central_mean, delta_t, mean_timediff, speed_ave


def calc_recovery_time(cpart_stat, frac, ref_frac):
    void_start_inds = np.where(frac - ref_frac > 0.7)[0]
    if len(void_start_inds) > 1:
        void_start_ind = void_start_inds[0]
    else:
        void_start_ind = 0

    void_end_ind = np.where(frac - ref_frac < 0.1)[0]
    potential_stop_inds = np.where(void_end_ind > void_start_ind)[0]
    if len(potential_stop_inds) > 1 and len(
            np.atleast_1d(void_start_inds)) > 1:
        void_end_ind = void_end_ind[potential_stop_inds[0]]
    else:
        void_start_ind = 0
        void_end_ind = 0
    start_void = cpart_stat.t_lims[void_start_ind]
    end_void = cpart_stat.t_lims[void_end_ind]

    recovery_time = end_void - start_void
    return recovery_time, start_void, end_void


def get_comp_scaling_data(
    canvas_parts_file="canvas_parts.pkl",
    canvas_parts_stats_file="canvas_part_stats_sw.pkl",
    filename="reddit_place_composition_list_extended_sw.csv",
    start_t_ind=12,
):
    """
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
    """

    with open(canvas_parts_file, "rb") as file:
        canvas_comp_list = pickle.load(file)

    with open(canvas_parts_stats_file, "rb") as file:
        canvas_part_stats_list = pickle.load(file)

    atlas, atlas_size = util.load_atlas()

    names = []
    subreddit = []
    size_pixels = []
    n_users_total = []

    n_defense_changes = []
    n_attack_changes = []
    n_defense_changes_start = []
    n_attack_changes_start = []
    n_defenseonly_users = []
    n_attackonly_users = []
    n_bothattdef_users = []
    n_defenseonly_users_start = []
    n_attackonly_users_start = []
    n_bothattdef_users_start = []

    n_ingroup_changes = []
    n_outgroup_changes = []
    n_ingroup_changes_start = []
    n_outgroup_changes_start = []
    n_ingrouponly_users = []
    n_outgrouponly_users = []
    n_bothinout_users = []
    n_ingrouponly_users_start = []
    n_outgrouponly_users_start = []
    n_bothinout_users_start = []

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
    start_black = []
    end_black = []
    recovery_black = []
    start_purple = []
    end_purple = []
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

        def_ch = np.sum(cpart_stat.n_defense_changes.val)
        att_ch = np.sum(cpart_stat.n_changes.val) - def_ch
        def_ch_st = np.sum(cpart_stat.n_defense_changes.val[0:start_t_ind])
        att_ch_st = np.sum(cpart_stat.n_changes.val[0:start_t_ind]) - def_ch_st

        in_ch = np.sum(cpart_stat.n_ingroup_changes.val)
        out_ch = np.sum(cpart_stat.n_changes.val) - in_ch
        in_ch_st = np.sum(cpart_stat.n_ingroup_changes.val[0:start_t_ind])
        out_ch_st = np.sum(cpart_stat.n_changes.val[0:start_t_ind]) - in_ch_st

        if len(canvas_comp.info.links
               ) != 0 and "subreddit" in canvas_comp.info.links:
            subreddit.append(canvas_comp.info.links["subreddit"])
        else:
            subreddit.append("NA")

        streamer_flag = 0
        if ("streamer" in canvas_comp.info.description) or (
                "stream" in canvas_comp.info.description):
            streamer_flag = 1

        alliance_flag = 0
        if (("alliance" in canvas_comp.info.description)
                or ("ally" in canvas_comp.info.description)
                or ("allies" in canvas_comp.info.description)):
            alliance_flag = 1
        elif subreddit[-1] != "NA" and (("Alliance" in canvas_comp.info.links)
                                        or ("ally" in canvas_comp.info.links)):
            alliance_flag = 1

        # dist_total, dist_mean, dist_central_mean, delta_t, speed_ave = calc_user_distance(canvas_comp)
        mean_attn, med_attn, pix_norm_attn = calc_user_attn(canvas_comp)

        recovery_time_black, start_t_black, end_t_black = calc_recovery_time(
            cpart_stat, cpart_stat.frac_black_px.val,
            cpart_stat.frac_black_ref.val)

        recovery_time_purple, start_t_purple, end_t_purple = calc_recovery_time(
            cpart_stat, cpart_stat.frac_purple_px.val,
            cpart_stat.frac_purple_ref.val)

        # add values to composition lists
        start_black.append(start_t_black)
        end_black.append(end_t_black)
        start_purple.append(start_t_purple)
        end_purple.append(end_t_purple)
        recovery_purple.append(recovery_time_purple)
        recovery_black.append(recovery_time_black)
        names.append(canvas_comp.info.atlasname)
        streamer.append(streamer_flag)
        alliance.append(alliance_flag)
        n_users_total.append(np.sum(cpart_stat.n_users_total))
        size_pixels.append(canvas_comp.num_pix())

        n_defense_changes.append(def_ch)
        n_attack_changes.append(att_ch)
        n_defense_changes_start.append(def_ch_st)
        n_attack_changes_start.append(att_ch_st)
        n_defenseonly_users.append(cpart_stat.n_defenseonly_users_lifetime)
        n_attackonly_users.append(cpart_stat.n_attackonly_users_lifetime)
        n_bothattdef_users.append(cpart_stat.n_bothattdef_users_lifetime)
        n_defenseonly_users_start.append(np.mean(cpart_stat.n_defenseonly_users.val[1:start_t_ind]))
        n_attackonly_users_start.append(np.mean(cpart_stat.n_attackonly_users.val[1:start_t_ind]))
        n_bothattdef_users_start.append(np.mean(cpart_stat.n_bothattdef_users.val[1:6]))

        n_ingroup_changes.append(in_ch)
        n_outgroup_changes.append(out_ch)
        n_ingroup_changes_start.append(in_ch_st)
        n_outgroup_changes_start.append(out_ch_st)
        n_ingrouponly_users.append(cpart_stat.n_ingrouponly_users_lifetime)
        n_outgrouponly_users.append(cpart_stat.n_outgrouponly_users_lifetime)
        n_bothinout_users.append(cpart_stat.n_bothinout_users_lifetime)
        n_ingrouponly_users_start.append(np.mean(cpart_stat.n_ingrouponly_users.val[1:start_t_ind]))
        n_outgrouponly_users_start.append(np.mean(cpart_stat.n_outgrouponly_users.val[1:start_t_ind]))
        n_bothinout_users_start.append(np.mean(cpart_stat.n_bothinout_users.val[1:start_t_ind]))

        tmin.append(cpart_stat.tmin)
        tmax.append(cpart_stat.tmax)
        tmin_quad.append(canvas_comp.tmin_quadrant())
        stability.append(cpart_stat.stability[0].val)  # may be an array
        instab.append(np.mean(
            cpart_stat.instability_norm[0].val))  # may be an array
        compressed_size.append(
            cpart_stat.size_compr_stab_im.val)  # may be an array
        entropy.append(cpart_stat.entropy_stab_im.val)  # may be an array
        fractal_dim.append(
            cpart_stat.fractal_dim_weighted.val)  # may be an array
        mean_attns.append(mean_attn)
        med_attns.append(med_attn)
        pix_norm_attns.append(pix_norm_attn)
        # dist_totals.append(dist_total)
        # dist_means.append(dist_mean)
        # dist_central_means.append(dist_central_mean)
        # delta_ts.append(delta_t)
        # speed_aves.append(speed_ave)

    data = {
        "Name": names,
        "Subreddit": subreddit,
        "Size (pixels)": size_pixels,
        "Num users total": n_users_total,
        "Num defense-only users": n_defenseonly_users,
        "Num attack-only users": n_attackonly_users,
        "Num attack-defense users": n_bothattdef_users,
        "Num defense-only users start": n_defenseonly_users_start,
        "Num attack-only users start": n_attackonly_users_start,
        "Num attack-defense users start": n_bothattdef_users_start,
        "Num defense changes": n_defense_changes,
        "Num attack changes": n_attack_changes,
        "Num defense changes start": n_defense_changes_start,
        "Num attack changes start": n_attack_changes_start,
        
        "Num ingroup changes": n_ingroup_changes,
        "Num outgroup changes": n_outgroup_changes,
        "Num ingroup changes start": n_ingroup_changes_start,
        "Num outgroup changes start": n_outgroup_changes_start,
        "Num ingrouponly users": n_ingrouponly_users,
        "Num outgrouponly users": n_outgrouponly_users,
        "Num bothinout users": n_bothinout_users,
        "Num ingroup-only users start": n_ingrouponly_users_start,
        "Num outgroup-only users start": n_outgrouponly_users_start,
        "Num bothinout users start": n_bothinout_users_start,

        "Instability": instab,
        "Streamer": streamer,
        "Alliance": alliance,
        "Start time (s)": tmin,
        "End time (s)": tmax,
        "Start time quadrant (s)": tmin_quad,
        "Stability": stability,
        "Compressed size": compressed_size,
        "Entropy": entropy,
        "End time (black)": end_black,
        "End time (purple)": end_purple,
        "Start time (black)": start_black,
        "Start time (purple)": start_purple,
        "Recovery time (black)": recovery_black,
        "Recovery time (purple)": recovery_purple,
        "Mean Attention": mean_attns,
        "Median Attention": med_attns,
        "Pixel Norm Attention": pix_norm_attns,
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    return df


def load_comp_scaling_data(
        filename="reddit_place_composition_list_extended.csv"):
    """
    load the dataframe of scaling data and split between different community types
    """
    df = pd.read_csv(filename)

    df_s = df[df["Streamer"] == 1]
    df_a = df[df["Alliance"] == 1]
    df_o = df[(df["Alliance"] == 0) & (df["Streamer"] == 0)]

    return df, df_s, df_a, df_o


def get_survivorship(df,
                     binning_var,
                     size_mins,
                     size_maxs,
                     times,
                     start_time,
                     norm=True,
                     plot=True):
    if plot:
        plt.figure()
    cmap = plt.get_cmap("copper")  # You can use any other colormap
    num_colors = len(size_mins)  # Change this to get more or fewer colors
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    survive_curves = []
    for k in range(len(size_mins)):
        num_per_time = np.zeros(len(times))
        df_size_bin = df[(binning_var > size_mins[k])
                         & (binning_var < size_maxs[k])
                         & (df["Start time quadrant (s)"] == start_time)]
        for i in range(len(df_size_bin)):
            t_end = np.array(df_size_bin["End time (s)"])[i]
            if t_end >= var.TIME_WHITEOUT:
                t_end = 10000000 * 3600
            t_start = np.array(df_size_bin["Start time (s)"])[i]
            t_survive = t_end - t_start
            for j in range(len(times)):
                if times[j] < t_survive:
                    num_per_time[j] += 1
        if norm:
            surv = num_per_time / num_per_time[0]
        else:
            surv = num_per_time
        if plot:
            plt.plot(
                times / 3600,
                surv,
                label=str(size_mins[k]) + "-" + str(size_maxs[k]),
                color=colors[k],
            )
        survive_curves.append(surv)
    if plot:
        plt.xlim(0, times[-1] / 3600)
        sns.despine()
        plt.xlabel("age (hrs)")
        plt.ylabel("number surviving")
    return times, survive_curves


def exponential_decay(x, A, tau, C):
    return A * np.exp(-x / tau) + C


def get_taus(df,
             binning_var,
             sizes,
             start_time,
             norm=False,
             plot=True,
             plot_surv=True):
    size_mins = sizes[0:-1]
    size_maxs = sizes[1:]
    times = np.arange(0, var.TIME_WHITEOUT - start_time + 30 * 60, 30 * 60)
    times, survive_curves = get_survivorship(
        df,
        binning_var,
        size_mins,
        size_maxs,
        times,
        start_time,
        norm=norm,
        plot=plot_surv,
    )

    if plot:
        plt.figure()
    num = len(survive_curves)
    taus = np.zeros(num)
    for i in range(num):
        initial_guess = (
            max(survive_curves[i]),
            10000,
            min(survive_curves[i]),
        )  # Initial guess for (A, tau, C)
        params, covariance = scipy.optimize.curve_fit(exponential_decay,
                                                      times,
                                                      survive_curves[i],
                                                      p0=initial_guess)
        A_fit, tau_fit, C_fit = params
        taus[i] = tau_fit
        if plot:
            plt.plot(times, survive_curves[i])
            plt.plot(
                times,
                exponential_decay(times, A_fit, tau_fit, C_fit),
                "--",
                label="Fit",
            )
    return taus
