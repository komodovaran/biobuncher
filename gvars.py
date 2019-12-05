import numpy as np
import sklearn.model_selection

def _simulate_traces(length=50, n_each_class=200):
    """
    Make 3 types of sequence data with variable length
    """
    data = []
    for _ in range(n_each_class):
        r = np.random.normal
        l = np.linspace
        i = np.random.randint

        x_noisy = np.column_stack(
            (
                (np.cos(l(i(1, 5), 5, length)) + r(0, 0.2, length)),
                ((1 + np.sin(l(i(1, 20), 5, length)) + r(0, 0.2, length))),
            )
        )

        x_wavy = np.column_stack(
            (
                (np.cos(l(0, i(1, 5), length)) + r(0, 0.2, length)),
                ((2 + np.sin(l(i(1, 20), 20, length)) + r(0, 0.2, length))),
            )
        )

        x_spikes = np.column_stack(
            (
                (np.cos(l(i(1, 5), 20, length)) + r(0, 0.2, length)) ** 3,
                (
                    (1 + np.sin(l(i(1, 20), 20, length)) + r(0, 0.2, length))
                    ** 3
                ),
            )
        )

        # Randomly cut the begining of traces and fill in with zeroes to mimick short traces
        zero = np.random.randint(1, length // 2)
        # x_noisy[0:zero] = 0
        # x_wavy[0:zero] = 0
        # x_spikes[0:zero] = 0

        data.append(x_noisy)
        data.append(x_wavy)
        data.append(x_spikes)

    data = np.array(data)
    data = data.reshape((-1, length, 2))

    X_train, X_test = sklearn.model_selection.train_test_split(
        data, train_size=0.8
    )

    mu = np.mean(X_train, axis=(0, 1))
    sg = np.std(X_train, axis=(0, 1))
    X_train = (X_train - mu) / sg
    X_test = (X_test - mu) / sg

    return X_train, X_test