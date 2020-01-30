import numpy as np
import matplotlib.pyplot as plt
import math
import lib.math
import parmap.parmap

def timeseries_to_gramian_angular_field(serie):
    """
    Converts univariate timeseries to gramian angular field

    https://medium.com/analytics-vidhya/encoding-time-series-as-images-b043becbdbf3
    """

    def _tabulate(x, y, f):
        """Return a table of f(x, y). Useful for the Gram-like operations."""
        return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

    def _cos_sum(a, b):
        """To work with tabulate."""
        return math.cos(a + b)

    # Min-Max scaling
    min_ = np.amin(serie)
    max_ = np.amax(serie)
    scaled_serie = (2 * serie - max_ - min_) / (max_ - min_)

    # Floating point inaccuracy!
    scaled_serie = np.where(scaled_serie >= 1.0, 1.0, scaled_serie)
    scaled_serie = np.where(scaled_serie <= -1.0, -1.0, scaled_serie)

    # Polar encoding
    phi = np.arccos(scaled_serie)

    # GAF Computation (every term of the matrix)
    gaf = _tabulate(x=phi, y=phi, f=_cos_sum)

    return gaf


if __name__ == "__main__":
    X = np.load("data/preprocessed/fake_tracks_type_3.npz", allow_pickle = True)["data"]

    def process(xi):
        ts = lib.math.resample_timeseries(xi, new_length = 32)
        img = np.dstack([timeseries_to_gramian_angular_field(ts[..., c]) for c in range(xi.shape[-1])])
        return img

    fig, ax = plt.subplots(nrows = 2)
    ax[0].plot(X[0])
    ax[1].plot(lib.math.resample_timeseries(X[0], new_length = 32))
    plt.show()

    mp_results = parmap.map(process, X)
    data = np.array(mp_results)
    print(data.shape)

    np.savez("data/preprocessed/fake_tracks_type_3_img.npz", data = data)