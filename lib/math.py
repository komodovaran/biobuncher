import numpy as np


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
    Normalizes each sample in a tensor to max value. Can be done for all features,
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
    return x / x.max(axis = (0, 1))