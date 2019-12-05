import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
from tensorflow.keras.models import Model

from lib.models import lstm_vae
from lib.tfcustom import AnnealingVariableCallback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_timeseries_data(length=50, n_each_class=200):
    """
    Make 3 types of sequence data with variable length
    """
    data = []
    labels = []
    for _ in range(n_each_class):
        r = np.random.normal
        l = np.linspace
        i = np.random.randint

        y_noise = 0
        x_noisy = np.column_stack(
            (
                (np.cos(l(i(1, 5), 5, length)) + r(0, 0.2, length)),
                ((1 + np.sin(l(i(1, 20), 5, length)) + r(0, 0.2, length))),
            )
        )

        y_wavy = 1
        x_wavy = np.column_stack(
            (
                (np.cos(l(0, i(1, 5), length)) + r(0, 0.2, length)),
                ((2 + np.sin(l(i(1, 20), 20, length)) + r(0, 0.2, length))),
            )
        )

        y_spikes = 2
        x_spikes = np.column_stack(
            (
                (np.cos(l(i(1, 5), 20, length)) + r(0, 0.2, length)) ** 3,
                (
                    (1 + np.sin(l(i(1, 20), 20, length)) + r(0, 0.2, length))
                    ** 3
                ),
            )
        )

        data.extend([x_noisy, x_wavy, x_spikes])
        labels.extend([y_noise, y_wavy, y_spikes])

    data = np.array(data)
    data = data.reshape((-1, length, 2))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data, labels, train_size=0.8
    )

    mu = np.mean(X_train, axis=(0, 1))
    sg = np.std(X_train, axis=(0, 1))
    X_train = (X_train - mu) / sg
    X_test = (X_test - mu) / sg

    return (X_train, X_test), (y_train, y_test)


if __name__ == "__main__":
    BATCH_SIZE = 64
    EPOCHS = 20

    (x_train, x_test), (y_train, y_test) = get_timeseries_data(
        length=200, n_each_class=500
    )

    vae = lstm_vae(
        n_timesteps=x_train.shape[1],
        n_features=x_train.shape[2],
        intermediate_dim=64,
        z_dim=2,
    )

    vae.fit(
        x=x_train,
        y=None,
        shuffle=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, None),)
    #     callbacks=[
    #         AnnealingVariableCallback(
    #             var =kl_weight, anneal_over_n_epochs=10, anneals_starts_at=10
    #         )
    #     ],
    # )

    encoder = Model(vae.input, vae.get_layer("z_sample").output)

    x_pred = vae.predict(x_test)

    fig, axes = plt.subplots(ncols=2, nrows=10, figsize=(10, 15))
    axes = axes.ravel()

    c1 = ["red", "green", "blue"]
    c2 = ["salmon", "lightgreen", "lightblue"]

    for i, ax in enumerate(axes):
        yi = y_test[i]
        ax.set_title(y_test[i])
        ax.plot(x_test[i], color=c1[yi], alpha=0.3)
        ax.plot(x_pred[i], color=c1[yi])

    manifold = encoder.predict(x_test)

    # Plot the manifold points
    fig, ax = plt.subplots()
    ax.scatter(manifold[:, 0], manifold[:, 1], c=y_test)
    plt.show()
