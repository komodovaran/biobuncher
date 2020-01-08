import numpy as np
import matplotlib.pyplot as plt
from lib.math import resample_timeseries


def fake_tracks_type_1():
    data = []
    for i in range(500):
        length = np.random.randint(low = 50, high = 200)

        clath = [1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 6, 5, 4, 2, 1]
        auxln = [1, 1, 1, 1, 2, 5, 2, 1, 2, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1]

        x = np.column_stack((clath, auxln))
        x = resample_timeseries(x, new_length = length)

        noise = np.random.normal(1, 0.2, len(x))

        # Add noise
        x[:, 0] *= noise
        x[:, 1] *= noise

        # Add intensities
        x[:, 0] *= np.random.uniform(300, 900)
        x[:, 1] *= np.random.uniform(300, 600)

        data.append(x)

    data = np.array(data)
    np.savez("results/intensities/fake_tracks_type_1.npz", data = data)
    return data

def fake_tracks_type_2():
    data = []
    for i in range(500):
        length = np.random.randint(low = 50, high = 200)

        clath = [1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 6, 5, 4, 2, 1]
        auxln = [1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        x = np.column_stack((clath, auxln))
        x = resample_timeseries(x, new_length = length)

        noise = np.random.normal(1, 0.2, len(x))

        # Add noise
        x[:, 0] *= noise
        x[:, 1] *= noise

        # Add intensities
        x[:, 0] *= np.random.uniform(300, 900)
        x[:, 1] *= np.random.uniform(300, 600)

        data.append(x)

    data = np.array(data)
    np.savez("results/intensities/fake_tracks_type_2.npz", data = data)
    return data


# data = fake_tracks_type_1()
data = fake_tracks_type_2()

fig, ax = plt.subplots(nrows = 4, ncols = 4)
ax = ax.ravel()
for i in range(len(ax)):
    ax[i].plot(data[i])
plt.show()