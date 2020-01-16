import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature
from collections import namedtuple
import skimage.io
import matplotlib.patches
from lib.utils import timeit
import skimage.filters
import warnings
from scipy.optimize import OptimizeWarning


def zoom_array(array, xy, square_radius):
    """
    Return a zoomed array at location
    """
    x, y = xy
    minix = int(max(x - square_radius, 0))
    miniy = int(max(y - square_radius, 0))
    maxix = int(x + square_radius)
    maxiy = int(y + square_radius)
    return array[miniy:maxiy, minix:maxix]


def gaussian_2d(
    xy_array, amplitude, pos_x, pos_y, sigma_x, sigma_y, angle, offset
):
    """
    Expression for a 2D gaussian function with variance in both x and y
    """
    x, y = xy_array

    a = (np.cos(angle) ** 2) / (2 * sigma_x ** 2) + (np.sin(angle) ** 2) / (
        2 * sigma_y ** 2
    )
    b = -(np.sin(2 * angle)) / (4 * sigma_x ** 2) + (np.sin(2 * angle)) / (
        4 * sigma_y ** 2
    )
    c = (np.sin(angle) ** 2) / (2 * sigma_x ** 2) + (np.cos(angle) ** 2) / (
        2 * sigma_y ** 2
    )

    g = offset + amplitude * np.exp(
        -(
            a * ((x - pos_x) ** 2)
            + 2 * b * (x - pos_x) * (y - pos_y)
            + c * ((y - pos_y) ** 2)
        )
    )
    return g.ravel()


def fit_gaussian_spots(x_guess, y_guess, array):
    Params = namedtuple(
        "Parameters", "amp, x, y, sigma_x, sigma_y, angle, offset"
    )

    initial_guess = Params(
        amp=np.max(array),
        x=x_guess,
        y=y_guess,
        sigma_x=1,
        sigma_y=1,
        angle=0,
        offset=np.abs(np.min(array)),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        try:
            X, Y = create_grid(*array.shape)
            popt, pcov = opt.curve_fit(
                f=gaussian_2d,
                xdata=(X, Y),
                ydata=array.ravel(),
                p0=initial_guess,
                maxfev=200,
                method="lm"
                # constraints make it slower. Better to time out bad fits
                # bounds=(min_bounds, max_bounds),
            )
            popt = Params(*np.round(popt))
        except (OptimizeWarning, ValueError, RuntimeError):
            popt, pcov = None, None
    return popt, pcov


def create_grid(h, w):
    """
    Creates a grid of x and y points to fit and evaluate over
    """
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    x, y = np.meshgrid(x, y)
    return x, y


def evaluate_gaussian(x, y, popt):
    """
    Evaluates gaussian in coordinate positions.
    NOTE: this is not necessary for extracting intensity,
    as the pure signal is fitted as the amplitude.
    """
    z = gaussian_2d((x, y), *popt)
    return z


if __name__ == "__main__":
    # Create x and y indices
    np.random.seed(4)
    PLOT_ROWS = 3
    PLOT_COLS = 3
    SIMULATE_IMAGE = False

    if SIMULATE_IMAGE:
        H, W = 200, 400
        Xi, Yi = create_grid(h=H, w=W)
        n_spots = 500

        # create tom_data
        img = []
        for _ in range(n_spots):
            POSX = np.random.randint(0, W)
            POSY = np.random.randint(0, H)
            AMP = 100
            g = gaussian_2d(
                xy_array=(Xi, Yi),
                amplitude=AMP,
                pos_x=POSX,
                pos_y=POSY,
                sigma_x=1,
                sigma_y=1,
                angle=0,
                offset=0,
            )
            img.append(g)
        img = np.sum(img, axis=0)
        img = img.reshape(H, W)
        img = img + np.random.normal(5, 5, len(img.ravel())).reshape(img.shape)
    else:
        img = skimage.io.imread(
            "../tom_data/A_CLTA-TagRFP EGFP-Aux1-A7D2/Cell1_1s/TagRFP/GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif"
        )
        img = img.mean(axis = 0) # get first frame

        Xi, Yi = create_grid(*img.shape)
    # Detect some possible spots first, and make some errors (distance too high)
    spots = skimage.feature.peak_local_max(img, num_peaks=300, min_distance=5)

    figimg, aximg = plt.subplots()
    aximg.imshow(
        img, origin="bottom", extent=(Xi.min(), Xi.max(), Yi.min(), Yi.max())
    )

    figzoom, axzoom = plt.subplots(nrows=PLOT_ROWS, ncols=PLOT_COLS)
    axzoom = axzoom.ravel()

    zoom_ctr = 6
    # skimage returns rows, columns (y,x) while matplotlib operates in (x,y)
    idx = 0
    for guessy, guessx in spots:
        # Plot on the full iamge
        # Initial
        aximg.add_patch(
            plt.Circle(
                (guessx, guessy), 3, linewidth=0.5, fill=False, color="yellow"
            )
        )

        # Fit
        local_arr = zoom_array(img, (guessx, guessy), square_radius=zoom_ctr)
        popt, pcov = fit_gaussian_spots(
            x_guess=zoom_ctr, y_guess=zoom_ctr, array=local_arr
        )
        if popt is None:
            continue

        predx = guessx + popt.x - zoom_ctr
        predy = guessy + popt.y - zoom_ctr

        # Plot on each of zooms
        # Predicted
        try:
            axzoom[idx].imshow(local_arr, origin = "bottom")
            axzoom[idx].add_patch(
                matplotlib.patches.Ellipse(
                    (popt.x, popt.y),
                    width=popt.sigma_x * 3,
                    height=popt.sigma_y * 3,
                    angle=popt.angle,
                    color="red",
                    fill=False,
                )
            )
            axzoom[idx].set_title(
                "fit: {:.1f} + {:.1f}\n"
                "est: {:.1f} + {:.1f}".format(
                    popt.amp, popt.offset, np.max(local_arr), np.min(local_arr)
                )
            )
        except IndexError:
            pass

        # Predicted
        aximg.add_patch(
            plt.Circle(
                (predx, predy), 3, linewidth=0.5, fill=False, color="green"
            )
        )

        idx += 1

    plt.tight_layout()
    plt.show()
