import numpy as np
import sklearn.model_selection
import streamlit as st
import tensorflow.python as tf

import lib.math
import lib.models


@st.cache
def _get_data(path):
    """
    Loads all traces
    """
    X = np.load(path)["data"]
    X = X[:, :, [0, 1]]

    return X


def _prepare_data(X, train_size=0.8):
    X_train, X_test = sklearn.model_selection.train_test_split(
        X, train_size=train_size, random_state=1
    )
    X_train_len, X_test_len = len(X_train), len(X_test)

    X_train, X_test = [
        tf.data.Dataset.from_tensor_slices((tf.constant(Xi), tf.constant(Xo)))
        for (Xi, Xo) in ((X_train, X_train), (X_test, X_test))
    ]

    X_train, X_test = [
        data.shuffle(buffer_size=10 * BATCH_SIZE).batch(BATCH_SIZE)
        for data in (X_train, X_test)
    ]
    return (X_train, X_test), (X_train_len, X_test_len)


if __name__ == "__main__":
    INPUT_DIR = "results/intensities/tracks-cme_split-c0c1_resampled-medianlen.npz"
    MODEL_DIR = None
    TAG = "DilatedConv"

    X_raw = _get_data(INPUT_DIR)
    X = lib.math.normalize_tensor(X_raw, feature_wise=False)

    N_TIMESTEPS = X.shape[1]
    N_FEATURES = X.shape[2]
    CALLBACK_TIMEOUT = 5
    EPOCHS = 100
    BATCH_SIZE = 32
    LATENT_DIM = 10

    (X_train, X_test), (X_train_len, X_test_len) = _prepare_data(X)

    model, callbacks, initial_epoch = lib.models.model_builder(
        model_dir=MODEL_DIR,
        chkpt_tag = TAG,
        model_build_f=lib.models.build_simple_conv_autoencoder,
        build_args=(N_FEATURES, LATENT_DIM, N_TIMESTEPS),
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