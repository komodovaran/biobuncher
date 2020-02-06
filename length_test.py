import numpy as np
import matplotlib.pyplot as plt
from lib.math import resample_timeseries

def generate_fake_tracks(savename):
    """The nth class will always be no signal"""
    data = []

    for l in [50, 100, 150]:
        for i in range(1000):
            clath = [1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 6, 5, 4, 2, 1]
            auxln = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 2, 1, 1, 1]
            x = np.column_stack((clath, auxln))

            # initial length
            x = resample_timeseries(x, new_length=l)

            noise = np.random.normal(1, 0.2, len(x))

            # Add noise
            x[:, 0] *= noise
            x[:, 1] *= noise

            # Add intensities
            x[:, 0] *= np.random.uniform(300, 900)
            x[:, 1] *= np.random.uniform(300, 600)


            # fix length again
            if l != 100:
                x = resample_timeseries(x, new_length = 100)

            data.append(x)

    data = np.array(data)
    np.savez("data/preprocessed/{}.npz".format(savename), data=data)
    return data

def main(savename):
    data = generate_fake_tracks(savename = savename)
    fig, ax = plt.subplots(nrows=4, ncols=4)
    ax = ax.ravel()
    for i in range(len(ax)):
        ax[i].plot(data[i])
    plt.show()


if __name__ == "__main__":
    main(savename = "length_test_resampled")
