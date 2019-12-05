import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf
import lib.math
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
    Preprocess data into x_y and appropriate train/test sets
    """
    X_train, X_test = sklearn.model_selection.train_test_split(
        X, train_size=train_size, random_state=0, shuffle=False
    )

    # axis 0 for both colums individually
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
    fig, ax = plt.subplots(nrows=4, ncols=4)
    fig.suptitle("Before scaling")
    ax = ax.ravel()
    for n in range(len((ax))):
        ax[n].plot(X_test[n])
    plt.show()

    # Apply scaler (standardization is the only thing that seems to work)
    X_train, X_test = [
        lib.math.standardize(X, mu, sg) for X in (X_train, X_test)
    ]

    # After scaling
    fig, ax = plt.subplots(nrows=4, ncols=4)
    fig.suptitle("After scaling")
    ax = ax.ravel()
    for n in range(len((ax))):
        ax[n].plot(X_test[n])
    plt.show()

    # Prepare dataset
    data, lengths = [], []
    for X in X_train, X_test:
        X_len = len(X)
        X = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(X)))
        X = X.shuffle(X_len).batch(BATCH_SIZE)
        data.append(X)
        lengths.append(X_len)

    return data, lengths


if __name__ == "__main__":
    EARLY_STOPPING = 20
    EPOCHS = 1000
    BATCH_SIZE = 128
    N_TIMESTEPS = 300
    N_FEATURES = 2

    CONTINUE_DIR = None
    MODELF = lib.models.lstm_autoencoder
    INPUT_NPZ = "results/intensities/tracks-cme_split-c1_res.npz"

    _LATENT_DIM = (32, 64, 128)
    _ACTIVATION = ("relu", "selu", "elu", "tanh", None)

    for (_latent_dim, _activation) in product(_LATENT_DIM, _ACTIVATION):

        X_raw = _get_data(INPUT_NPZ)

        (X_train, X_test), (X_train_len, X_test_len) = _preprocess(
            X_raw, path=INPUT_NPZ,
        )

        build_args = [N_TIMESTEPS, N_FEATURES, _latent_dim, _activation]

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
