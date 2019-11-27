import numpy as np
import parmap
import scipy.interpolate
from scipy import fftpack
from lib.utils import timeit, est_proc
import sklearn.mixture


def div0(a, b):
    """Converts all zero-division results to zeros"""
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c


def smooth_spline_2d(image, **spline_kwargs):
    """
    Fits a smooth spline in 2D, for image smoothing.
    See RectBivariateSpline for additional options.
    """
    y, x = image.shape
    rows = np.arange(0, y, 1)
    cols = np.arange(0, x, 1)
    return scipy.interpolate.RectBivariateSpline(rows, cols, image, **spline_kwargs)(rows, cols)


def circle_mask(inner_area, outer_area, gap_space, yx, indices):
    """
    Calculates a circular pixel mask for extracting intensities

    Parameters
    ----------
    inner_area:
        Area of the inner ROI
    outer_area:
        Area of ROI + background + space
    gap_space:
        Area of a circle
    yx:
        Coordinates in the format (y, x) (i.e. matrix indices row/col)
    indices:
        Image indices (obtained from np.indices(img.shape))

    Returns
    -------
    Center and background ring ROIs
    """

    yy, xx = yx

    yi, xi = indices
    mask = (yy - yi) ** 2 + (xx - xi) ** 2

    center = mask <= inner_area ** 2
    gap = mask <= inner_area ** 2 + gap_space ** 2
    bg_filled = mask <= outer_area ** 2
    # subtract inner circle_overlap from outer
    bg_ring = np.logical_xor(bg_filled, gap)

    return center, bg_ring


def frame_roi_intensity(array, roi_mask, bg_mask):
    """
    Extracts get_intensities from TIFF stack, given ROI and BG masks.
    Intensities are calculated as medians of all pixel values within the ROIs.

    Parameters
    ----------
    array:
        Single-channel currentMovie array
    roi_mask:
        Numpy mask for center
    bg_mask:
        Numpy mask for background
    bg_mode:
        Type of background estimation to use

    Returns
    -------
    Corrected intensity
    """
    if not len(array.shape) == 2:
        raise ValueError("Only works on single-channel frames")

    # Count the number of signal pixels
    roi_n_pixels = np.sum(roi_mask)

    # whole ROI integrated (sum of all pixels)
    roi_pixel_sum_intensity = np.sum(array[roi_mask])

    # median background intensity (single pixel)
    bg_pixel_intensity = np.median(array[bg_mask])

    # Subtract background value from every pixel
    corrected_intensity = roi_pixel_sum_intensity - (
        bg_pixel_intensity * roi_n_pixels
    )

    # Get average signal pixel intensity
    mean_corrected_intensity = corrected_intensity / roi_n_pixels

    return mean_corrected_intensity


def calc_steplength(df, x_col, y_col):
    """
    Gets the calc_steplength for every step in (x,y) using Euclidian distance.
    The function returns inplace
    """
    df[["dif_x", "dif_y"]] = (
        df[[x_col, y_col]]
            .rolling(window = 2)
            .apply(lambda row: row[1] - row[0], raw = True)
    )
    df["steplength"] = np.sqrt(df["dif_x"] ** 2 + df["dif_y"] ** 2)
    df["steplength"].replace(np.nan, 0, inplace = True)
    df.drop(["dif_x", "dif_y"], axis = 1, inplace = True)
    return df["steplength"]


def maxabs_tensor(X, per_feature = False):
    """
    Sample-wise max-value normalization of 3D array (tensor).
    This is not feature-wise normalization, to keep the ratios between extracted_features intact!
    """
    if len(X.shape) == 2:
        X = X[np.newaxis, :, :]
    if not len(X.shape) == 3:
        raise ValueError("Shape not a tensor")

    if per_feature:
        axis = 1
    else:
        axis = (1, 2)
    arr_max = np.max(X, axis = axis, keepdims = True)

    X = div0(X, arr_max)
    return np.squeeze(X)


def znorm_tensor(X, per_feature):
    if len(X.shape) == 2:
        X = X[np.newaxis, :, :]
    if not len(X.shape) == 3:
        raise ValueError("Shape not a tensor")

    if per_feature:
        axis = 1
    else:
        axis = (1, 2)
    arr_mean = np.mean(X, axis = axis, keepdims = True)
    arr_std = np.mean(X, axis = axis, keepdims = True)
    return np.squeeze((X - arr_mean) / arr_std)


def z_score_norm(x):
    """
    Z-score normalizes array
    """
    return (x - np.mean(x)) / np.std(x)


def maxabs_norm(x):
    """
    Maxabs normalizes array
    """
    return x / x.max(axis = (0, 1))


def fit_gaussian_mixture(arr, k_states):
    """
    Fits k gaussians to a set of data.
    Parameters
    ----------
    arr:
        Input data (wil be unravelled to single-sample shape)
    k_states:
        Maximum number of states to test for:
    Returns
    -------
    Parameters zipped as (means, sigmas, weights), BICs and best k if found by BIC method
    Returned as a dictionary to avoid unpacking the wrong things when having few parameters
    Examples
    --------
    # For plotting the returned parameters:
    for i, params in enumerate(gaussfit_params):
        m, s, w = params
        ax.plot(xpts, w * scipy.stats.norm.pdf(xpts, m, s))
        sum.append(np.array(w * stats.norm.pdf(xpts, m, s)))
    joint = np.sum(sum, axis = 0)
    ax.plot(xpts, joint, color = "black", alpha = 0.05)
    """
    if len(arr) < 2:
        return None, None

    arr = arr.reshape(-1, 1)

    bics_ = []
    gs_ = []
    best_k = None
    if type(k_states) == range:
        for k in k_states:
            g = sklearn.mixture.GaussianMixture(n_components = k)
            g.fit(arr)
            bic = g.bic(arr)
            gs_.append(g)
            bics_.append(bic)

        best_k = np.argmin(bics_).astype(int) + 1
        g = sklearn.mixture.GaussianMixture(n_components = best_k)
    else:
        g = sklearn.mixture.GaussianMixture(n_components = k_states)

    g.fit(arr)

    weights = g.weights_.ravel()
    means = g.means_.ravel()
    sigs = np.sqrt(g.covariances_.ravel())

    params = [(m, s, w) for m, s, w in zip(means, sigs, weights)]
    params = sorted(params, key = lambda tup: tup[0])

    return dict(params = params, bics = bics_, best_k = best_k)


def resample_timeseries(y, new_length = None):
    """
    Resamples timeseries by linear interpolation
    """
    ndim = y.shape[1]

    x = range(len(y))
    if new_length is None:
        new_length = len(x)

    new_x = np.linspace(min(x), max(x), new_length)
    new_y = np.zeros((new_length, ndim))

    # interpolate for each of the channels individually and collect
    for i in range(ndim):
        f = scipy.interpolate.interp1d(x, y[:, i])
        new_y[:, i] = f(new_x)
    return new_y


def peak_region_finder(y, lag = 30, threshold = 5, influence = 0):
    """
    Detects peak regions. See more at
    https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data

    For example, a lag of 5 will use
    the last 5 observations to smooth the data. A threshold of 3.5 will signal
    if a datapoint is 3.5 standard deviations away from the moving mean.
    And an influence of 0.5 gives signals half of the influence that normal
    datapoints have. Likewise, an influence of 0 ignores signals completely for
    recalculating the new threshold. An influence of 0 is therefore the most
    robust option (but assumes stationarity); putting the influence option at 1
    is least robust. For non-stationary data, the influence option should
    therefore be put somewhere between 0 and 1.

    Args:
        y:
            Input signal
        lag:
            The lag of the moving window
        threshold:
            The z-score at which the algorithm signals
        influence:
            The influence (between 0 and 1) of new signals on the mean and
            standard deviation

    Returns:
        Array containing the signal of a peak region
    """
    signals = np.zeros(len(y))
    filtered_y = np.array(y)
    avg_filter = [0] * len(y)
    std_filter = [0] * len(y)
    avg_filter[lag - 1] = np.mean(y[0:lag])
    std_filter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avg_filter[i - 1]) > threshold * std_filter[i - 1]:
            if y[i] > avg_filter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filtered_y[i] = influence * y[i] + (1 - influence) * filtered_y[i - 1]
            avg_filter[i] = np.mean(filtered_y[(i - lag + 1): i + 1])
            std_filter[i] = np.std(filtered_y[(i - lag + 1): i + 1])
        else:
            signals[i] = 0
            filtered_y[i] = y[i]
            avg_filter[i] = np.mean(filtered_y[(i - lag + 1): i + 1])
            std_filter[i] = np.std(filtered_y[(i - lag + 1): i + 1])

    return signals


def nd_fft_ts(y, center = False, log_transform = False):
    """
    Calculates fast fourier transform for each feature in a ND-timeseries
    """
    y = y.reshape(len(y), -1)
    if center:
        y = y - np.mean(y, axis = 0)  # center the signal to avoid zero component
    ndim = y.shape[1]
    new_y = np.zeros((y.shape[0] // 2, ndim))
    for i in range(ndim):
        yf = fftpack.fft(y[:, i])
        # Remove mirrored and negative parts
        yf_single = np.abs(yf[:len(y) // 2])
        # log transform may make it nicer visually
        if log_transform:
            yf_single = np.log(1 + yf_single)
        new_y[:, i] = yf_single
    return new_y


def fft_bg_2d(image, K = 2, percentile = 10):
    """
    Background correction with Fast Fourier Transform on a 2D image.
    Args:
        K:
            Block size set to zero. Higher number gives more "wiggliness".
        percentile:
            Percentile above which to filter out. Higher number adds more noise.
    """
    M, N = image.shape

    F = fftpack.fftn(image)
    F_magnitude = np.abs(F)
    F_magnitude = fftpack.fftshift(F_magnitude)

    # Set a block around center of spectrum to zero
    F_magnitude[M // 2 - K: M // 2 + K, N // 2 - K: N // 2 + K] = 0

    # Find all peaks higher than the 98th percentile
    peaks = F_magnitude < np.percentile(F_magnitude, percentile)

    # Shift the peaks back to align with the original spectrum
    peaks = fftpack.ifftshift(peaks)

    # Set those peak coefficients to zero
    F_dim = F * peaks

    # Do the inverse Fourier transform to get back to an image.
    # Since we started with a real image, we only look at the real part of
    # the output.
    image_filtered = np.real(fftpack.ifft2(F_dim))
    return image_filtered


@timeit
def fft_bg_video(video, K = 2, percentile = 10, return_subtracted = True):
    """
    Parallel implementation of fft_bg_2d for videos
    Return subtracted to immediately subtract the FFT (background)
    from the original
    """
    bg = np.array(parmap.map(fft_bg_2d, video, K, percentile, pm_processes = est_proc()))
    return video - bg if return_subtracted else bg


def mean_squared_error(A, B, axis):
    """Calculates mean squared error between two ndarrays on a given axis"""
    return ((A - B) ** 2).mean(axis = axis)


def ragged_stat(arr, f):
    """
    Returns the statistic of a ragged array, given a function
    """
    arr = np.array(arr)
    return f(np.concatenate(arr).ravel())