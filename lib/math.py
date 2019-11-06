import numpy as np
from scipy.interpolate import interp1d


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


def roi_intensity(array, roi_mask, bg_mask, bg_mode = "min"):
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
    Center and background get_intensities
    """
    if not len(array.shape) == 2:
        raise ValueError("Only works on single-channel frames")

    # whole ROI integrated (sum of all pixels)
    roi_pixel_sum_intensity = np.sum(array[roi_mask])

    # median background intensity (single pixel)
    if bg_mode == "median":
        bg_pixel_intensity = np.median(array[bg_mask])
    elif bg_mode == "min":
        bg_pixel_intensity = np.min(array[bg_mask])
    else:
        raise ValueError("Background mode must be 'median' or 'min'")

    # Count the number of signal pixels
    roi_n_pixels = np.sum(roi_mask)

    # Subtract background value from every pixel
    corrected_intensity = roi_pixel_sum_intensity - (
        bg_pixel_intensity * roi_n_pixels
    )

    # Get average signal pixel intensity
    mean_corrected_intensity = corrected_intensity / roi_n_pixels

    return mean_corrected_intensity, bg_pixel_intensity


def calc_steplength(df, x_col, y_col):
    """
    Gets the calc_steplength for every step in (x,y) using Euclidian distance.
    The function returns inplace
    """
    df[["dif_x", "dif_y"]] = (
        df[[x_col, y_col]]
        .rolling(window=2)
        .apply(lambda row: row[1] - row[0], raw=True)
    )
    df["steplength"] = np.sqrt(df["dif_x"] ** 2 + df["dif_y"] ** 2)
    df.drop(["dif_x", "dif_y"], axis=1, inplace=True)
    return df["steplength"]


def normalize_tensor(X_raw, feature_wise=False):
    """
    Normalizes each sample in a tensor to max value. Can be done for all extracted_features,
    or separately
    """
    if feature_wise:
        maxval = np.max(X_raw, axis=1, keepdims=True)
    else:
        maxval = np.max(X_raw, axis=(1, 2), keepdims=True)
    X = X_raw / maxval
    return X


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


def resample_timeseries(y, new_length = None):
    """
    Resamples timeseries by linear interpolation
    """
    xpts = range(len(y))
    f = interp1d(xpts, y)
    if new_length is None:
        new_length = len(xpts)
    newy = f(np.linspace(min(xpts), max(xpts), new_length))
    return newy


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