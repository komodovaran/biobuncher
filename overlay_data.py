import numpy as np
import matplotlib.pyplot as plt

def generate_fake_tracks():
    data = []
    for i in range(16):
        a = np.sin(np.arange(100) * 0.1)
        a = abs(a)

        b = np.ones(len(a))

        x = np.column_stack((a, b))

        noise = np.random.normal(2, 0.2, len(x))

        # Add noise
        x[:, 0] *= noise
        x[:, 1] *= noise

        # Add intensities
        x[:, 0] *= np.random.uniform(200, 1000)
        x[:, 1] *= np.random.uniform(100)

        data.append(x)

    data = np.array(data)
    return data


def main():
    data = generate_fake_tracks()
    fig, ax = plt.subplots(nrows=4, ncols=4)
    ax = ax.ravel()
    for i in range(len(ax)):
        ax[i].plot(data[i])
    plt.show()

    overlay = data.mean(axis = 0)
    fig, ax = plt.subplots()
    ax.plot(overlay)
    plt.show()

if __name__ == "__main__":
    main()
