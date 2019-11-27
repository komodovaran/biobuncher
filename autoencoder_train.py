import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf
import lib.models

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _get_data(path):
    """
    Loads all traces
    """
    print(path)
    X = np.load(path, allow_pickle=True)["data"]
    if X.shape[0] < 100:
        raise ValueError("File is suspiciously small. Recheck!")
    return X


def _preprocess(X, path, train_size=0.8):
    """
    Preprocess data into tensors and appropriate train/test sets
    """
    X_train, X_test = sklearn.model_selection.train_test_split(
        X, train_size = train_size, random_state = 0, shuffle = False
    )

    # axis 0 for both colums individually

    # make it easier to do stats on
    X_stat = np.row_stack(X_train)
    mu = np.mean(X_stat, axis=(0))
    sg = np.std(X_stat, axis=(0))

    # Save before applying scaler
    np.savez(
        path[:-4] + "_traintest.npz",
        X_train=X_train,
        X_test=X_test,
        scale=(mu, sg),
    )

    # Before scaling
    fig, ax = plt.subplots(nrows=5, ncols=5)
    fig.suptitle("Before scaling")
    ax = ax.ravel()
    for n in range(len((ax))):
        ax[n].plot(X_test[n])
    plt.show()

    # Apply scaler
    # Standardize (the only thing that works it seems)
    X_train = np.array([(xi - mu) / sg for xi in X_train])
    X_test = np.array([(xi - mu) / sg for xi in X_test])

    # After scaling
    fig, ax = plt.subplots(nrows=5, ncols=5)
    fig.suptitle("After scaling")
    ax = ax.ravel()
    for n in range(len((ax))):
        ax[n].plot(X_test[n])
    plt.show()

    # Fit
    X_train_len, X_test_len = len(X_train), len(X_test)

    if len(X.shape) > 1:
        X_train, X_test = [
            tf.data.Dataset.from_tensor_slices(
                (tf.constant(Xi), tf.constant(Xo))
            )
            for (Xi, Xo) in ((X_train, X_train), (X_test, X_test))
        ]
    else:
        X_train, X_test = [
            tf.data.Dataset.from_tensor_slices(
                (tf.constant(Xi), tf.constant(Xo))
            )
            for (Xi, Xo) in ((X_train, X_train), (X_test, X_test))
        ]

    X_train, X_test = [
        data.shuffle(buffer_size=10 * BATCH_SIZE).batch(BATCH_SIZE)
        for data in (X_train, X_test)
    ]
    return (X_train, X_test), (X_train_len, X_test_len)


if __name__ == "__main__":
    EARLY_STOPPING = 20
    EPOCHS = 1000
    BATCH_SIZE = 128
    N_TIMESTEPS = 300
    CONTINUE_DIR = None
    MODELF = lib.models.multi_lstm_autoencoder
    INPUT_NPZ = "results/intensities/tracks-cme_split-c1_res.npz"

    _LATENT_DIM = (32, 64, 128)
    _ACTIVATION = ("relu", "selu", "elu", "tanh", None)

    for (_latent_dim, _activation) in itertools.product(
        _LATENT_DIM, _ACTIVATION
    ):

        X_raw = _get_data(INPUT_NPZ)

        (X_train, X_test), (X_train_len, X_test_len) = _preprocess(
            X_raw, path=INPUT_NPZ,
        )

        n_timesteps = X_raw.shape[1]
        n_features = X_raw.shape[2]

        if re.search("lstm", MODELF.__name__):
            n_timesteps = 300
        build_args = [n_timesteps, n_features, _latent_dim, _activation]

        TAG = "_{}".format(MODELF.__name__)
        TAG += "_dim={}".format(_latent_dim)
        TAG += "_data={}".format(INPUT_NPZ.split("/")[-1])

        model, callbacks, initial_epoch = lib.models.model_builder(
            model_dir=CONTINUE_DIR,
            chkpt_tag=TAG,
            patience=EARLY_STOPPING,
            model_build_f=MODELF,
            build_args=build_args,
        )

        model.summary()
        model.fit(
            x=X_train.repeat(),
            validation_data=X_test.repeat(),
            epochs=EPOCHS,
            steps_per_epoch=X_train_len // BATCH_SIZE,
            validation_steps=X_test_len // BATCH_SIZE,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
        )
