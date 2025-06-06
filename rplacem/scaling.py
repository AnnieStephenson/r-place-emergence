import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import scipy.odr 
import piecewise_regression
import rplacem.canvas_part as cp
import rplacem.entropy as ent
import rplacem.utilities as util
from rplacem import var as var


def linear_model(params, x):
    return params[0] * x + params[1]

def plot_loglog_fit(x_data_unfilt,
                    y_data_unfilt,
                    z_data_unfilt=None,
                    x_data_filter_min=0,
                    y_data_filter_min=0,
                    z_data_filter_min=0,
                    cmap='viridis',
                    x_data_filter_max=None,
                    y_data_filter_max=None,
                    data_color=[0.3, 0.3, 0.3],
                    line_color=[0, 0, 0],
                    markersize=2,
                    markersize_bin=4,
                    alpha_data=0.15,
                    alpha_line=0.7,
                    alpha_error=0.3,
                    elinewidth=0,
                    linewidth=2.5,
                    nbins=None,
                    bin_type='average', # 'average'
                    bin_axis='y', # 'x'
                    fit_type='TLS', # 'OLS'
                    TLS_weights=False,
                    max_bin_size=10,
                    n_subsample_iter=100,
                    plot_sliding_window_ave=True,
                    x_line_max=None,
                    semilog=False, fit=True, 
                    print_summary=False,
                    plot_bin_data=False):
    """
    Plots the data on a loglog plot with linear fit

    Notes:
        Explanation of how the error bars are being calculated, when doing the bilinear fit

        1. We first bin the data and find the mean and standard deviation of the data in each bin. 
        2. The plotted line fit is the fit on the mean data. 
        3. Then, we fit the mean + stdev and means - stdev to get a high and low estimate 
           based on the variance of the data
        4. Then we use the confidence intervals on those high and low fits to obtain a high and low estimate based on
           both the variance in the data and the confidence on the fit. 
    """
    if linewidth == 0:
        alpha_error = 0

    x_data_unfilt = np.array(x_data_unfilt)
    y_data_unfilt = np.array(y_data_unfilt)
    if z_data_unfilt is not None:
        z_data_unfilt = np.array(z_data_unfilt)

    # filter
    x_data = x_data_unfilt[(x_data_unfilt > x_data_filter_min) & (y_data_unfilt > y_data_filter_min)]
    y_data = y_data_unfilt[(x_data_unfilt > x_data_filter_min) & (y_data_unfilt > y_data_filter_min)]
    if z_data_unfilt is not None:
        # [(z_data_unfilt > z_data_filter_min) & (z_data_unfilt > z_data_filter_min)]
        z_data = z_data_unfilt[(x_data_unfilt > x_data_filter_min) & (y_data_unfilt > y_data_filter_min)]

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
            cmap=cmap,
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

    # calculate unbinned log data for later use in R^2
    log_x_data_raw = np.log10(x_data)
    log_y_data_raw = np.log10(y_data)

    log_x_data, log_y_data, log_x_sem, log_y_sem = handle_data_bins(x_data, y_data, nbins, bin_type, bin_axis, max_bin_size=max_bin_size)
    
    if plot_bin_data:
        # Transform log-space means to linear
        x_mean = 10**log_x_data
        y_mean = 10**log_y_data

        # Compute asymmetric error bars
        if bin_axis == 'y':
            x_upper = 10**(log_x_data + log_x_sem)
            x_lower = 10**(log_x_data - log_x_sem)
            xerr = np.vstack([x_mean - x_lower, x_upper - x_mean])
            yerr = None
        elif bin_axis == 'x':
            y_upper = 10**(log_y_data + log_y_sem)
            y_lower = 10**(log_y_data - log_y_sem)
            yerr = np.vstack([y_mean - y_lower, y_upper - y_mean])
            xerr = None

        # Plot mean points
        plt.plot(x_mean, y_mean, '.', color=line_color, markersize=markersize_bin)

        # Plot error bars with asymmetric intervals
        plt.errorbar(
            x_mean, y_mean,
            xerr=xerr, yerr=yerr,
            ecolor=line_color, elinewidth=elinewidth,
            capsize=0, linewidth=0)

    if fit_type=='bilinear':
        pw_fit = piecewise_regression.Fit(log_x_data, log_y_data, n_breakpoints=1)
        results = pw_fit.get_results()
        intercept = results['estimates']['const']['estimate']
        slope = results['estimates']['alpha1']['estimate']
        slope2 = results['estimates']['alpha2']['estimate'] 
        breakpt = results['estimates']['breakpoint1']['estimate']
        pw_fit_low = piecewise_regression.Fit(log_x_data + 2*log_x_sem, log_y_data - 2*log_y_sem, n_breakpoints=1)
        pw_fit_high = piecewise_regression.Fit(log_x_data - 2*log_x_sem, log_y_data + 2*log_y_sem, n_breakpoints=1)

        intercept_conf = pw_fit.get_results()['estimates']['const']['confidence_interval']
        slope_conf = pw_fit.get_results()['estimates']['alpha1']['confidence_interval']
        slope2_conf = pw_fit.get_results()['estimates']['alpha2']['confidence_interval']
        breakpt_conf = pw_fit.get_results()['estimates']['breakpoint1']['confidence_interval']
       
        #intercept_conf = (pw_fit_low.get_results()['estimates']['const']['estimate'], pw_fit_high.get_results()['estimates']['const']['estimate'])
        #slope_conf = (pw_fit_low.get_results()['estimates']['alpha1']['estimate'], pw_fit_high.get_results()['estimates']['alpha1']['estimate'])
        #slope2_conf = (pw_fit_low.get_results()['estimates']['alpha2']['estimate'], pw_fit_high.get_results()['estimates']['alpha2']['estimate'])
        #breakpt_conf = (pw_fit_low.get_results()['estimates']['breakpoint1']['estimate'], pw_fit_high.get_results()['estimates']['breakpoint1']['estimate'])

        #intercept_conf = (pw_fit_low.get_results()['estimates']['const']['confidence_interval'][0], pw_fit_high.get_results()['estimates']['const']['confidence_interval'][1])
        #slope_conf = (pw_fit_low.get_results()['estimates']['alpha1']['confidence_interval'][0], pw_fit_high.get_results()['estimates']['alpha1']['confidence_interval'][1])
        #slope2_conf = (pw_fit_low.get_results()['estimates']['alpha2']['confidence_interval'][0], pw_fit_high.get_results()['estimates']['alpha2']['confidence_interval'][1])
        #breakpt_conf_low = (pw_fit_low.get_results()['estimates']['breakpoint1']['confidence_interval'][0], pw_fit_low.get_results()['estimates']['breakpoint1']['confidence_interval'][1])
        #breakpt_conf_high = (pw_fit_high.get_results()['estimates']['breakpoint1']['confidence_interval'][0], pw_fit_high.get_results()['estimates']['breakpoint1']['confidence_interval'][1])
        if print_summary:
            print('\nFit summary for the mean of binned data: ')
            pw_fit.summary()
            print('\nFit summary for the mean of binned data minus stdev: ')
            pw_fit_low.summary()
            print('\nFit summary for the mean of binned data plus stdev:')
            pw_fit_high.summary()
    if fit_type=='OLS':
        coefficients = np.polyfit(log_x_data, log_y_data, 1)
        slope, intercept = coefficients
        slope_conf = None
        intercept_conf = None
    if fit_type=='TLS':
        if TLS_weights:
            data = scipy.odr.RealData(log_x_data, log_y_data, sx=log_x_sem, sy=log_y_sem)
        else:
            data = scipy.odr.RealData(log_x_data, log_y_data)
        model = scipy.odr.Model(linear_model)
        odr = scipy.odr.ODR(data, model, beta0=[1.0, 0.0])  # Initial guess for parameters
        result = odr.run()
        params = result.beta
        slope= params[0]
        intercept = params[1]

        # calculate R**2 on binned data
        SST = np.sum((log_y_data - np.mean(log_y_data))**2)
        y_pred = linear_model(result.beta, log_x_data)
        SSR = np.sum((log_y_data - y_pred)**2)
        R_squared = 1 - (SSR / SST)
        RMS_error = np.sqrt(np.mean((log_y_data - y_pred)**2))
        print( 'R squared binned: ' + str(R_squared))
        print( 'RMS error binned: ' + str(RMS_error))
       

        # calculate R**2 on rawdata
        SST = np.sum((log_y_data_raw - np.mean(log_y_data_raw))**2)
        y_pred = linear_model(result.beta, log_x_data_raw)
        SSR = np.sum((log_y_data_raw - y_pred)**2)
        R_squared = 1 - (SSR / SST)
        RMS_error_raw = np.sqrt(np.mean((log_y_data_raw - y_pred)**2))
        print( 'R squared raw: ' + str(R_squared))
        print( 'RMS error raw: ' + str(RMS_error_raw))
        

        # Confidence level, e.g., 95%
        confidence_level = 0.95

        # Degrees of freedom = number of data points - number of parameters
        df = len(log_x_data) - len(result.beta)

        # Critical value from t-distribution
        t_val = scipy.stats.t.ppf((1 + confidence_level) / 2., df)

        # Confidence intervals
        conf_intervals = []
        for param, error in zip(result.beta, result.sd_beta):
            ci_width = t_val * error
            conf_intervals.append((param - ci_width, param + ci_width))
        slope_conf = conf_intervals[0]
        intercept_conf = conf_intervals[1]
    print('\n')
    print('Fit parameters and roughly estimateed confidence intervals: ')
    print("intercept: " + str(intercept) + ', conf. interval: ' + str(intercept_conf))
    print("scaling: " + str(10**intercept))
    print("exponent: " + str(slope) + ', conf. interval: ' + str(slope_conf))

    x_line_data = np.linspace(np.log10(np.min(x_data)), np.log10(np.max(x_data)), num=500)
    x_line_data_lin = 10 ** x_line_data
    if fit_type == 'bilinear':
        ind_bp = np.argmin(np.abs(breakpt-x_line_data))
        plt.plot(x_line_data_lin, 10 ** pw_fit.predict(x_line_data),
                 color=line_color,
                 alpha=alpha_line,
                 linewidth=linewidth)
        plt.vlines(x=10**breakpt, linestyle='--',
                   alpha=0.5,
                   color= line_color, 
                   ymin = np.min(y_data), 
                   ymax=np.max(y_data))
        
        # Print the parameters for the main fit
        y_fit1 = 10 ** intercept * x_line_data_lin[0:ind_bp] ** slope
        intercept2 = np.log10(y_fit1[-1]) - np.log10(x_line_data_lin[ind_bp] ** slope2)
        intercept2_low = np.log10(y_fit1[-1]) - np.log10(x_line_data_lin[ind_bp] ** slope2_conf[1])
        intercept2_high = np.log10(y_fit1[-1]) - np.log10(x_line_data_lin[ind_bp] ** slope2_conf[0])
        print("breakpoint: " + str(breakpt) + ', conf. interval: ' + str((breakpt_conf)))
        print("exponent 2: " + str(slope2) + ', conf. interval: ' + str(slope2_conf))
        print("intercept 2: " + str(intercept2) + ', conf. interval: ' + str((intercept2_low, intercept2_high)))

        # Calculate the y values for the high and low fit prediction
        # high end of low fit. Higher bkpt tends to give lower y value, hence taking the higher one for the low fit
        ind_bp_low = np.argmin(np.abs(breakpt_conf[1]-x_line_data)) 
        # low end of high fit. Lower bkpt tends to give higher y value, hence takig the lower one for the high fit
        ind_bp_high = np.argmin(np.abs(breakpt_conf[0]-x_line_data)) 
        
        yfit1_low = 10 ** intercept_conf[0] * x_line_data_lin[0:ind_bp_low] ** slope_conf[0]
        yfit1_high = 10 ** intercept_conf[1] * x_line_data_lin[0:ind_bp_high] ** slope_conf[1]
        intercept2_low = np.log10(yfit1_low[-1]) - np.log10(x_line_data_lin[ind_bp_low] ** slope2_conf[0])
        intercept2_high = np.log10(yfit1_high[-1]) - np.log10(x_line_data_lin[ind_bp_high] ** slope2_conf[1])

        yfit2_low = 10 ** intercept2_low * x_line_data_lin[ind_bp_low:] ** slope2_conf[0]
        yfit2_high = 10 ** intercept2_high * x_line_data_lin[ind_bp_high:] ** slope2_conf[1]

        yfit_low = np.concatenate([yfit1_low, yfit2_low])
        yfit_high = np.concatenate([yfit1_high, yfit2_high])

        plt.fill_between(
            x_line_data_lin, 
            yfit_low, 
            yfit_high, 
            linewidth=0,
            color=line_color, alpha=alpha_error)

    else:
        plt.plot(
            x_line_data_lin,
            10**(intercept) * x_line_data_lin**(slope),
            color=line_color,
            alpha=alpha_line,
            linewidth=linewidth,
        )
        if fit_type=='TLS':
            if linewidth > 0:
                yfit_low =  10 ** intercept_conf[0] * x_line_data_lin ** slope_conf[0]
                yfit_high =  10 ** intercept_conf[1] * x_line_data_lin ** slope_conf[1]
                plt.fill_between(x_line_data_lin, 
                                yfit_low, 
                                yfit_high, 
                                linewidth=0,
                                color=line_color, alpha=alpha_error)


    if z_data_unfilt is not None:
        data = x_data, y_data, z_data
    else:
        data = x_data, y_data
    return data

def handle_data_bins(x_data, y_data, nbins, bin_type, bin_axis, max_bin_size=10, bandwidth=0.4, grid_size=80):
    """
    Binning or smoothing of log-log data based on bin_type.
    
    Parameters
    ----------
    x_data : array
        Raw x data.
    y_data : array
        Raw y data.
    nbins : int
        Number of bins (ignored if bin_type='kernel_average').
    bin_type : str
        'median', 'average', 'subsample', or 'kernel_average'.
    bin_axis : str
        'x' or 'y', depending which variable you want to bin along.
    max_bin_size : int
        Maximum subsample size (for 'subsample' mode).
    bandwidth : float
        Kernel bandwidth (only used if bin_type='kernel_average').
    grid_size : int
        Number of points in the kernel smoothed curve (only for 'kernel_average').
        
    Returns
    -------
    result : list of arrays
        [x_values, y_values, x_sem, y_sem] after binning or smoothing.
    """
    if bin_axis == 'y':
        log_x = np.log10(x_data)
        log_y = np.log10(y_data)
    elif bin_axis == 'x':
        log_x = np.log10(y_data)
        log_y = np.log10(x_data)
    else:
        raise ValueError("bin_axis must be 'x' or 'y'")

    if bin_type == 'kernel_average':
        # Smooth log-log data using the kernel_smooth_xy()
        x_grid, x_smooth, y_smooth, x_sem, y_sem = kernel_smooth_xy(log_y, log_x, bandwidth=bandwidth, grid_size=grid_size)
        
        # x_smooth and y_smooth already correspond to smoothed x and y, with SEMs
        y_vals = x_smooth
        x_vals = y_smooth
        y_sem_vals = x_sem
        x_sem_vals = y_sem

    else:
        # Regular binning (median, average, or subsample)
        hist_values, bin_edges = np.histogram(log_y, bins=nbins)

        x_vals = np.array([])
        y_vals = np.array([])
        x_sem_vals = np.array([])
        y_sem_vals = np.array([])

        for i in range(len(bin_edges) - 1):
            indices = np.where((log_y >= bin_edges[i])
                               & (log_y < bin_edges[i + 1])
                               & (~np.isnan(log_x)))[0]
            if len(indices) < 2:
                continue  # Skip empty or single-point bins

            if bin_type == 'median':
                x_bin = np.array([np.median(log_x[indices])])
                y_bin = np.array([np.median(log_y[indices])])
                x_sem = np.array([scipy.stats.sem(log_x[indices], ddof=1)])
                y_sem = np.array([scipy.stats.sem(log_y[indices], ddof=1)])
            elif bin_type == 'average':
                x_bin = np.array([np.mean(log_x[indices])])
                y_bin = np.array([np.mean(log_y[indices])])
                x_sem = np.array([scipy.stats.sem(log_x[indices], ddof=1)])
                y_sem = np.array([scipy.stats.sem(log_y[indices], ddof=1)])
            elif bin_type == 'subsample':
                max_samp = min(len(indices), max_bin_size)
                sampled_inds = np.random.choice(indices, size=max_samp, replace=False)
                x_bin = log_x[sampled_inds]
                y_bin = log_y[sampled_inds]
                x_sem = np.zeros_like(x_bin)
                y_sem = np.zeros_like(y_bin)
            else:
                raise ValueError(f"Unknown bin_type: {bin_type}")

            x_vals = np.concatenate((x_vals, x_bin))
            y_vals = np.concatenate((y_vals, y_bin))
            x_sem_vals = np.concatenate((x_sem_vals, x_sem))
            y_sem_vals = np.concatenate((y_sem_vals, y_sem))

    # Flip back depending on bin_axis
    if bin_axis == 'y':
        result = [x_vals, y_vals, x_sem_vals, y_sem_vals]
    else:  # bin_axis == 'x'
        result = [y_vals, x_vals, y_sem_vals, x_sem_vals]

    return result

def kernel_smooth_xy(x, y, bandwidth=0.4, grid_size=100, weight_threshold_factor = 0.0005):
    """
    Kernel smoothing for both x and y with SEM estimation.
    
    Parameters
    ----------
    x : 1D array
        Input x values (may be noisy).
    y : 1D array
        Input y values.
    bandwidth : float
        Kernel bandwidth for smoothing.
    grid_size : int
        Number of points in the smoothed curve.
        
    Returns
    -------
    x_grid : 1D array
        Grid points where smoothing was evaluated (requested).
    x_smooth : 1D array
        Weighted average x values (smoothed x).
    y_smooth : 1D array
        Weighted average y values (smoothed y).
    x_sem : 1D array
        Standard error of x at each grid point.
    y_sem : 1D array
        Standard error of y at each grid point.
    """
    
    weight_threshold = weight_threshold_factor * len(x)

    x = np.asarray(x)
    y = np.asarray(y)
    
    x_grid = np.linspace(np.min(x), np.max(x), grid_size)
    
    distances = (x[:, None] - x_grid[None, :]) / bandwidth
    weights = scipy.stats.norm.pdf(distances)
    
    sum_weights = np.sum(weights, axis=0)
    
    # Weighted means
    weighted_x_mean = np.sum(weights * x[:, None], axis=0) / sum_weights
    weighted_y_mean = np.sum(weights * y[:, None], axis=0) / sum_weights

    # Weighted variances
    weighted_var_x = np.sum(weights * (x[:, None] - weighted_x_mean[None, :])**2, axis=0) / sum_weights
    weighted_var_y = np.sum(weights * (y[:, None] - weighted_y_mean[None, :])**2, axis=0) / sum_weights
    
    # Effective sample size
    eff_n = (sum_weights**2) / np.sum(weights**2, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sem_x = np.sqrt(weighted_var_x / eff_n)
        sem_y = np.sqrt(weighted_var_y / eff_n)

    # Mask out low-support points
    valid = sum_weights > weight_threshold
    x_grid = x_grid[valid]
    x_smooth = weighted_x_mean[valid]
    y_smooth = weighted_y_mean[valid]
    sem_x = sem_x[valid]
    sem_y = sem_y[valid]

    return x_grid, x_smooth, y_smooth, sem_x, sem_y



def handle_data_bins_old(x_data, y_data, nbins, bin_type, bin_axis, max_bin_size=10):
    if bin_axis=='y':
        log_x_data = np.log10(x_data)
        log_y_data = np.log10(y_data)
    if bin_axis=='x':
        log_x_data = np.log10(y_data)
        log_y_data = np.log10(x_data)
    hist_values, bin_edges = np.histogram(log_y_data, bins=nbins)

    # Get the values within each bin
    y_log_values_clip = np.array([])
    x_log_values_clip = np.array([])
    y_log_sem_vals = np.array([])
    x_log_sem_vals = np.array([])

    for i in range(len(bin_edges) - 1):
        indices = np.where((log_y_data >= bin_edges[i])
                           & (log_y_data < bin_edges[i + 1])
                           & (~np.isnan(log_x_data)))[0]
        if bin_type=='median':
            y_log_values = np.array([np.median(log_y_data[indices])])
            x_log_values = np.array([np.median(log_x_data[indices])])
            y_log_sem = np.array([scipy.stats.sem(log_y_data[indices], ddof=1)])
            x_log_sem = np.array([scipy.stats.sem(log_x_data[indices], ddof=1)])       
        if bin_type=='average':
            y_log_values = np.array([np.mean(log_y_data[indices])])
            x_log_values = np.array([np.mean(log_x_data[indices])])
            y_log_sem = np.array([np.std(log_y_data[indices])]) #np.array([scipy.stats.sem(log_y_data[indices], ddof=1)])
            x_log_sem = np.array([np.std(log_x_data[indices])]) #np.array([scipy.stats.sem(log_x_data[indices], ddof=1)])            
        if bin_type=='subsample':
            if len(indices) < max_bin_size:
                max_samp = len(indices)
            else:
                max_samp = max_bin_size
            sampled_inds = np.random.choice(indices, size=max_samp)
            x_log_values = log_x_data[sampled_inds]
            y_log_values = log_y_data[sampled_inds]
            y_log_sem = np.array([0])
            x_log_sem = np.array([0])
        if not np.isnan(x_log_values[0]) and not np.isnan(y_log_sem[0]) and len(indices)>1:
            x_log_values_clip = np.concatenate((x_log_values_clip, x_log_values))
            y_log_values_clip = np.concatenate((y_log_values_clip, y_log_values))
            y_log_sem_vals = np.concatenate((y_log_sem_vals, y_log_sem))
            x_log_sem_vals = np.concatenate((x_log_sem_vals, x_log_sem))
    if bin_axis=='y':
        result = [x_log_values_clip, y_log_values_clip, x_log_sem_vals, y_log_sem_vals]
    if bin_axis=='x':
        result = [y_log_values_clip, x_log_values_clip, y_log_sem_vals, x_log_sem_vals]

    return result


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
import numpy as np

def calc_user_distance(canvas_comp):
    """
    Compute spatial‑and‑temporal activity statistics for every **active** user
    in a `CanvasComposition`.

    Parameters
    ----------
    canvas_comp : CanvasComposition
        Object describing a single composition on the r/place canvas.  This
        function relies on three public attributes:

        pixel_changes : structured ndarray
            A record array with at least the fields

            * ``"user"``      – integer or hash identifying the author
            * ``"coord_index"`` – index into ``canvas_comp.coords`` giving the
              pixel edited
            * ``"seconds"``   – Unix time (or seconds since epoch  / start)
            * ``"active"``    – 1 = include in analysis, 0 = ignore
        pixch_sortuser : ndarray of int
            Permutation that orders ``pixel_changes`` first by *user* then by
            time (monotonic within each user).  It allows all edits by a given
            user to be contiguous in the array, which makes the per‑user
            slicing below O(1).
        coords : (2, N) ndarray of float
            Pixel coordinate look‑up table where column *i* contains the
            *(x, y)* position of pixel index *i* on the canvas.

    Returns
    -------
    dist_total : (U,) ndarray
        **Total path length** each user’s cursor travelled, computed as the sum
        of Euclidean distances between *consecutive* pixel edits.
        ``np.nan`` when the user edited only a single pixel.
    dist_mean : (U,) ndarray
        **Average step length** between consecutive edits for each user.
        ``np.nan`` for single‑edit users.
    dist_central_mean : (U,) ndarray
        **Mean distance to centroid** – the average Euclidean distance of every
        edited pixel from that user’s *own* centre of activity
        ``(x̄, ȳ)``.  Low values imply a compact working region.
    delta_t : (U,) ndarray
        **Session duration** per user in seconds (last edit time minus first
        edit time).  Zero when the user has exactly one active edit.
    mean_timediff : (U,) ndarray
        **Mean inter‑edit time** for each user, i.e. the average of
        ``np.diff(times)``.  ``np.nan`` for single‑edit users.
    speed_ave : (U,) ndarray
        **Effective drawing speed** in pixels ⋅ s⁻¹, defined as
        ``dist_total / delta_t``.  ``np.nan`` whenever `dist_total` is
        ``np.nan`` *or* `delta_t` is zero.

    Notes
    -----
    * Only edits with ``"active" == 1`` are analysed.
    * Users with no active edits are automatically excluded, so the length *U*
      of every returned array equals the number of active users.
    """

    # Keep only active pixel changes and place them in "all‑edits‑by‑user" order
    pixch_user_sorted = canvas_comp.pixel_changes[canvas_comp.pixch_sortuser]
    pixch_user_sorted = pixch_user_sorted[pixch_user_sorted["active"] == 1]

    # `users` gives the unique user IDs in *reverse* order (because of [::-1]),
    # `user_inds` are the indices of each user's *last* edit in the sorted array
    users, user_inds = np.unique(pixch_user_sorted["user"][::-1],
                                 return_index=True)
    # Convert indices of last edits into lengths of each user slice
    user_end_inds = len(pixch_user_sorted) - user_inds           # shape (U,)

    # Pre‑allocate result arrays (float64 by default, filled with zeros)
    U = len(users)
    dist_total        = np.zeros(U)
    dist_mean         = np.zeros(U)
    dist_central_mean = np.zeros(U)
    delta_t           = np.zeros(U)
    mean_timediff     = np.zeros(U)
    speed_ave         = np.zeros(U)

    # -------------------------------------------------------------
    # Iterate over users.  This loop is fast in practice (< 10⁴ users),
    # and keeps the code easy to read; full vectorisation would add
    # complexity for little speed‑up.
    # -------------------------------------------------------------
    for j in range(U):
        # Determine the slice of rows belonging to the *j*‑th user
        start = 0 if j == 0 else user_end_inds[j - 1]
        stop  = user_end_inds[j]
        coord_inds = pixch_user_sorted["coord_index"][start:stop]
        times      = pixch_user_sorted["seconds"][start:stop]

        # ------------------------------------------------------------------
        # Temporal statistics
        # ------------------------------------------------------------------
        delta_t[j] = times[-1] - times[0]              # session length
        if len(times) > 1:
            mean_timediff[j] = np.mean(np.diff(times))
        else:
            mean_timediff[j] = np.nan                  # undefined for one edit

        # ------------------------------------------------------------------
        # Spatial statistics
        # ------------------------------------------------------------------
        coords_x, coords_y = canvas_comp.coords[:, coord_inds]

        # 1. Mean distance to user's own centroid
        x_mean = coords_x.mean()
        y_mean = coords_y.mean()
        dist_central_mean[j] = np.mean(
            np.hypot(coords_x - x_mean, coords_y - y_mean)
        )

        # 2. Distances between *sequential* edits (cursor path length)
        if len(coord_inds) > 1:
            dist_steps = np.hypot(                     # vectorised hypot √(dx²+dy²)
                np.diff(coords_x),
                np.diff(coords_y)
            )
            dist_total[j] = dist_steps.sum()
            dist_mean[j]  = dist_steps.mean()
        else:
            dist_total[j] = dist_mean[j] = np.nan      # only one edit ⇒ no steps

        # ------------------------------------------------------------------
        # Average speed (guard against division by zero or nan)
        # ------------------------------------------------------------------
        speed_ave[j] = (dist_total[j] / delta_t[j]
                        if delta_t[j] > 0 and not np.isnan(dist_total[j])
                        else np.nan)

    return (dist_total,
            dist_mean,
            dist_central_mean,
            delta_t,
            mean_timediff,
            speed_ave)




def calc_user_distance_old(canvas_comp):
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
    canvas_parts_stats_file="canvas_part_stats_sw.pkl",
    canvas_parts_file=None,
    filename="reddit_place_composition_list_extended_sw.csv",
    start_t_ind=12, # 12 five-min increments = 1 hour
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
    print(start_t_ind)
    if canvas_parts_file is not None:
        with open(canvas_parts_file, "rb") as file:
            canvas_comp_list = pickle.load(file)
        print(len(canvas_comp_list))

    with open(canvas_parts_stats_file, "rb") as file:
        canvas_part_stats_list = pickle.load(file)

    print(len(canvas_part_stats_list))
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
    n_users_total_start = []

    instab = []
    streamer = []
    flag = []
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

    for i in range(num_iter):
        print(i)

        cpart_stat = canvas_part_stats_list[i]
        #canvas_comp = canvas_comp_list[i]

        if cpart_stat.n_defense_changes.val is None:
            def_ch = 0
            def_ch_st =0
        else:
            def_ch = np.sum(cpart_stat.n_defense_changes.val)
            def_ch_st = np.sum(cpart_stat.n_defense_changes.val[0:start_t_ind])
        att_ch = np.sum(cpart_stat.n_changes.val) - def_ch
        att_ch_st = np.sum(cpart_stat.n_changes.val[0:start_t_ind]) - def_ch_st

        in_ch = np.sum(cpart_stat.n_ingroup_changes.val)
        out_ch = np.sum(cpart_stat.n_changes.val) - in_ch
        in_ch_st = np.sum(cpart_stat.n_ingroup_changes.val[0:start_t_ind])
        out_ch_st = np.sum(cpart_stat.n_changes.val[0:start_t_ind]) - in_ch_st

        if len(cpart_stat.info.links
               ) != 0 and "subreddit" in cpart_stat.info.links:
            subreddit.append(cpart_stat.info.links["subreddit"])
        else:
            subreddit.append("NA")

        flag_flag = 0
        if ("flag" in cpart_stat.info.description):
            flag_flag = 1

        streamer_flag = 0
        if ("streamer" in cpart_stat.info.description) or (
                "stream" in cpart_stat.info.description):
            streamer_flag = 1

        alliance_flag = 0
        if (("alliance" in cpart_stat.info.description)
                or ("ally" in cpart_stat.info.description)
                or ("allies" in cpart_stat.info.description)):
            alliance_flag = 1
        elif subreddit[-1] != "NA" and (("Alliance" in cpart_stat.info.links)
                                        or ("ally" in cpart_stat.info.links)):
            alliance_flag = 1

        if canvas_parts_file is not None:
            print(canvas_comp_list[i].info.atlasname)
            print(cpart_stat.info.atlasname)
            dist_total, dist_mean, dist_central_mean, delta_t, mean_timediff, speed_ave = calc_user_distance(canvas_comp_list[i])
            #mean_attn, med_attn, pix_norm_attn = calc_user_attn(canvas_comp)

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
        names.append(cpart_stat.info.atlasname)
        streamer.append(streamer_flag)
        flag.append(flag_flag)
        alliance.append(alliance_flag)
        n_users_total.append(np.sum(cpart_stat.n_users_total))
        size_pixels.append(cpart_stat.area)

        n_defense_changes.append(def_ch)
        n_attack_changes.append(att_ch)
        n_defense_changes_start.append(def_ch_st)
        n_attack_changes_start.append(att_ch_st)
        n_defenseonly_users.append(cpart_stat.n_defenseonly_users_lifetime)
        n_attackonly_users.append(cpart_stat.n_attackonly_users_lifetime)
        n_bothattdef_users.append(cpart_stat.n_bothattdef_users_lifetime)
        if n_defenseonly_users[i] is None:
            n_defenseonly_users[i] = 0
        if n_attackonly_users[i] is None:
            n_attackonly_users[i] = 0
        if n_bothattdef_users[i] is None:
            n_bothattdef_users[i] = 0

        f_new = cpart_stat.frac_users_new_vs_previoustime.val[1:start_t_ind]
        n_users_total_start.append(np.sum(f_new*cpart_stat.n_users.val[1:start_t_ind]))
        if cpart_stat.n_defenseonly_users.val is None:
            n_defenseonly_users_start.append(0)
            n_attackonly_users_start.append(0)
            n_bothattdef_users_start.append(0)
        else:
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
        #tmin_quad.append(canvas_comp.tmin_quadrant())
        stability.append(cpart_stat.stability[0].val)  # may be an array
        instab.append(np.mean(cpart_stat.instability_norm[0].val))  # may be an array
        if cpart_stat.size_compressed.val is None:
            compressed_size.append(0)
        else:
            compressed_size.append(np.mean(cpart_stat.size_compressed.val))  # may be an array
        entropy.append(np.mean(cpart_stat.entropy.val))  # may be an array
        fractal_dim.append(np.mean(cpart_stat.fractal_dim_weighted.val))  # may be an array
        #mean_attns.append(mean_attn)
        #med_attns.append(med_attn)
        #pix_norm_attns.append(pix_norm_attn)
        if canvas_parts_file is not None:
            dist_totals.append(np.nanmean(dist_total))
            dist_means.append(np.nanmean(dist_mean))
            dist_central_means.append(np.nanmean(dist_central_mean))
            delta_ts.append(np.nanmean(delta_t))
            speed_aves.append(np.nanmean(speed_ave))
        else:
            dist_totals.append(np.nan)
            dist_means.append(np.nan)
            dist_central_means.append(np.nan)
            delta_ts.append(np.nan)
            speed_aves.append(np.nan)

    # a few more vars
    lifetimes = np.array(tmax) - np.array(tmin)
    #if canvas_parts_stats_file[-1]=='3':
    lifetimes_percent = lifetimes / (var.TIME_WHITEOUT - np.array(tmin))
    num_changes_tot = np.array(n_defense_changes) + np.array(n_attack_changes)
    num_users_tot = np.array(n_defenseonly_users) + np.array(n_attackonly_users) + np.array(n_bothattdef_users)
    user_density = num_users_tot / np.array(size_pixels) / lifetimes

    data = {
        "Name": names,
        "Subreddit": subreddit,
        "Size (pixels)": size_pixels,
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
        'Lifetimes': lifetimes,
        'Lifetimes percent': lifetimes_percent,
        'Num changes tot': num_changes_tot,
        'Num users tot': num_users_tot,
        'User density': user_density,
        "Instability": instab,
        "Streamer": streamer,
        "Flag": flag,
        "Alliance": alliance,
        "Start time (s)": tmin,
        "End time (s)": tmax,
        #"Start time quadrant (s)": tmin_quad,
        "Stability": stability,
        "Compressed size": compressed_size,
        "Entropy": entropy,
        "Fractal dimension": fractal_dim,
        "End time (black)": end_black,
        "End time (purple)": end_purple,
        "Start time (black)": start_black,
        "Start time (purple)": start_purple,
        "Recovery time (black)": recovery_black,
        "Recovery time (purple)": recovery_purple,
        "dist_total": dist_totals,
        "dist_mean": dist_means,
        "dist_central_mean": dist_central_means,
        "delta_t": delta_ts,
        "speed_ave": speed_aves,
        'Num users total start': n_users_total_start
        #"Mean Attention": mean_attns,
        #"Median Attention": med_attns,
        #"Pixel Norm Attention": pix_norm_attns,
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    return df

def get_time_scaling_data(
    canvas_parts_stats_file="canvas_part_stats_sw.pkl",
    filename="n_changes_vst_vs_comp.npz", 
    n_skip=1):

    """
   
    """

    with open(canvas_parts_stats_file, "rb") as file:
        canvas_part_stats_list = pickle.load(file)

    num_iter = len(canvas_part_stats_list)
    num_times = int(np.ceil(((var.TIME_WHITEOUT / canvas_part_stats_list[0].t_unit) + 1) / n_skip))

    # metric of interest
    n_changes_vst_vs_comp = np.zeros((num_iter, num_times))
    n_in_ch_vst_vs_comp = np.zeros((num_iter, num_times))
    n_out_ch_vst_vs_comp = np.zeros((num_iter, num_times))
    print(num_times)
   
    for i in range(num_iter):
        print(i)
        cpart_stat = canvas_part_stats_list[i]

        n_changes_vst_vs_comp[i, :len(cpart_stat.n_changes.val)] = cpart_stat.n_changes.val
        n_in_ch_vst_vs_comp[i, :len(cpart_stat.n_changes.val)] = cpart_stat.n_ingroup_changes.val
        n_out_ch_vst_vs_comp[i, :len(cpart_stat.n_changes.val)] = cpart_stat.n_outgroup_changes.val

    np.savez(filename, n_changes_vst_vs_comp, n_in_ch_vst_vs_comp, n_out_ch_vst_vs_comp)

    return n_changes_vst_vs_comp, n_in_ch_vst_vs_comp, n_out_ch_vst_vs_comp


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
                     plot=True,
                     extend_lifetimes=False):
    if plot:
        plt.figure()
    cmap = plt.get_cmap("copper")  # You can use any other colormap
    num_colors = len(size_mins)  # Change this to get more or fewer colors
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    survive_curves = []
    df_list = []
    for k in range(len(size_mins)):
        num_per_time = np.zeros(len(times))
        if extend_lifetimes:
            # take all of the compositions in size bins
            df_size_bin = df[(binning_var > size_mins[k])
                            & (binning_var < size_maxs[k])
                            & (df["Start time quadrant (s)"] == start_time)]
        else:
            # include only compositions that end before the experiment ends. 
            df_size_bin = df[(binning_var > size_mins[k])
                            & (binning_var < size_maxs[k])
                            & (df["Start time quadrant (s)"] == start_time)
                            & (df["End time (s)"] < var.TIME_WHITEOUT)]
            
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
        df_list.append(df_size_bin)
        survive_curves.append(surv)
    if plot:
        plt.xlim(0, times[-1] / 3600)
        sns.despine()
        plt.xlabel("age (hrs)")
        plt.ylabel("number surviving")
    return times, survive_curves, df_list


def exponential_decay(x, A, tau):
    return A * np.exp(-x / tau)# + C


def get_taus(df,
             binning_var,
             sizes,
             start_time,
             norm=False,
             plot=True,
             plot_surv=True,
             extend_lifetimes=False):
    
    size_mins = sizes[0:-1]
    size_maxs = sizes[1:]
    times = np.arange(0, var.TIME_WHITEOUT - start_time - 10 * 30 * 60, 30 * 60)
    times, survive_curves, df_list = get_survivorship(df, binning_var,
                                             size_mins,
                                             size_maxs,
                                             times,
                                             start_time,
                                             norm=norm,
                                             plot=plot_surv,
                                             extend_lifetimes=extend_lifetimes)

    if plot:
        plt.figure()
    num = len(survive_curves)
    taus = np.zeros(num)
    for i in range(num):
        initial_guess = (
            max(survive_curves[i]),
            10000)
            #min(survive_curves[i])) # Initial guess for (A, tau, C)
        params, covariance = scipy.optimize.curve_fit(exponential_decay,
                                                      times,
                                                      survive_curves[i],
                                                      p0=initial_guess)
        A_fit, tau_fit = params
       # print(C_fit)
        taus[i] = tau_fit
        if plot:
            plt.semilogy(times, survive_curves[i])
            plt.plot(
                times,
                exponential_decay(times, A_fit, tau_fit),
                "--",
                label="Fit",
            )
    return taus, df_list
