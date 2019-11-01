import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature
from collections import namedtuple
import skimage.io
from lmfit import Parameters, minimize, report_fit


def gaussian_2d(x, y, cen_x, cen_y, sig_x, sig_y, offset):
    return np.exp(-(((cen_x - x) / sig_x) ** 2 + ((cen_y - y) / sig_y) ** 2) / 2.0) + offset


def gaussian_residuals(p, x, y, z):
    height = p["height"].value
    cen_x = p["centroid_x"].value
    cen_y = p["centroid_y"].value
    sigma_x = p["sigma_x"].value
    sigma_y = p["sigma_y"].value
    offset = p["background"].value
    return (z - height * gaussian_2d(x, y, cen_x, cen_y, sigma_x, sigma_y, offset))


def create_grid(h, w):
    """
    Creates a grid of x and y points to fit and evaluate over
    """
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    x, y = np.meshgrid(x, y)
    return x, y


if __name__ == "__main__":
    # Create x and y indices
    np.random.seed(4)
    h, w = 200, 200
    x, y = create_grid(h = h, w = w)

    # create data
    img = []
    for _ in range(10):
        # sx = np.random.uniform(0, 3)
        # sy = np.random.uniform(0, 3)
        posx = np.random.randint(10, w - 10)
        posy = np.random.randint(10, h - 10)
        g = 10 * gaussian_2d(x, y, posx, posy, 3, 3, 0)
        img.append(g)

    img = np.sum(img, axis = 0)
    img = img.reshape(h, w)
    img += np.random.normal(0, 0.1, len(img.ravel())).reshape(img.shape)

    print("max intensity: {:.2f}".format(img.max()))

    # Detect soem possible spots first
    spots = skimage.feature.peak_local_max(img, num_peaks = 10, min_distance = 10)
    fig, ax = plt.subplots(ncols = 2)

    h, w = img.shape
    local_area = 20

    # skimage returns rows, columns (y,x) while matplotlib operates in (x,y)
    fits = []
    for idx, (pre_y, pre_x) in enumerate(spots):
        initial = Parameters()
        initial.add("height", value = 1.)
        initial.add("centroid_x", value = 1)
        initial.add("centroid_y", value = 1)
        initial.add("sigma_x", value = 3.)
        initial.add("sigma_y", value = 3.)
        initial.add("background", value = 0.)

        minx = int(max(pre_x - local_area, 0))
        miny = int(max(pre_y - local_area, 0))
        maxx = int(pre_x + local_area)
        maxy = int(pre_y + local_area)
        lcl = img[miny: maxy, minx: maxx]

        _x, _y = create_grid(*lcl.shape)

        fit = minimize(gaussian_residuals, initial, args = (_x, _y, lcl))
        n, *popt = np.array(fit.params)

        fits.append(gaussian_2d(x, y, pre_x, pre_y, *popt[2:]))

    fits = np.sum(fits, axis = 0)

    ax[0].set_title("true")
    ax[0].imshow(
        img, origin = "bottom", extent = (x.min(), x.max(), y.min(), y.max())
    )
    ax[1].set_title("fits")
    ax[1].imshow(fits, origin = "bottom")

    plt.show()
