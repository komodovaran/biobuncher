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


def roi_intensity(array, roi_mask, bg_mask):
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
    raw:
        Whether to return raw signal/background get_intensities. Otherwise will return signal-background and background as zeroes.

    Returns
    -------
    Center and background get_intensities
    """
    if not len(array.shape) == 2:
        raise ValueError("Only works on single-channel frames")

    # whole ROI integrated (sum of all pixels)
    roi_pixel_sum_intensity = np.sum(array[roi_mask])

    # median background intensity (single pixel)
    median_bg_pixel_intensity = np.median(array[bg_mask])

    # Count the number of signal pixels
    roi_n_pixels = np.sum(roi_mask)

    # Subtract background value from every pixel
    corrected_intensity = roi_pixel_sum_intensity - (
        median_bg_pixel_intensity * roi_n_pixels
    )

    # Get average signal pixel intensity
    mean_corrected_intensity = corrected_intensity / roi_n_pixels

    return mean_corrected_intensity, median_bg_pixel_intensity