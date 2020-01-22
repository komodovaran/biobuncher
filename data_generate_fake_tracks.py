import numpy as np
import matplotlib.pyplot as plt
from lib.math import resample_timeseries

def generate_fake_tracks(n_classes=3):
    """The nth class will always be no signal"""
    data = []
    for i in range(16):
        length = np.random.randint(low=50, high=200)

        clath = [1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 6, 5, 4, 2, 1]
        auxln = [1] * len(clath)

        possible_pos = np.linspace(0, len(auxln), n_classes - 1, dtype=int)
        pos = np.random.choice(possible_pos)
        idx = pos + np.random.randint(-2, 2)
        x = np.column_stack((clath, auxln))

        x = resample_timeseries(x, new_length=length)

        noise = np.random.normal(1, 0.2, len(x))

        x[idx : idx + 30, 1] += 3

        # Add noise
        x[:, 0] *= noise
        x[:, 1] *= noise

        # Add intensities
        x[:, 0] *= np.random.uniform(300, 900)
        x[:, 1] *= np.random.uniform(300, 600)

        data.append(x)

    data = np.array(data)
    np.savez("data/preprocessed/fake_tracks.npz", data=data)
    return data


def main():
    data = generate_fake_tracks()
    fig, ax = plt.subplots(nrows=4, ncols=4)
    ax = ax.ravel()
    for i in range(len(ax)):
        ax[i].plot(data[i])
    plt.show()


if __name__ == "__main__":
    main()
