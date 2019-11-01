import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import streamlit as st
import lib.math
import lib.plotting
import lib.utils
import lib.models
import time

import tensorflow as tf

from lib.models import model_builder


def _plot_examples(X_raw):
    fig, ax = plt.subplots(nrows=4, ncols=4)
    ax = ax.ravel()
    rand_idx = np.random.randint(0, len(X), 16).tolist()

    for i, r in enumerate(rand_idx):
        xi, = lib.utils.remove_zero_padding(X_raw[r, ...])
        ax[i].plot(xi[:, 0], color="salmon")
        ax[i].plot(xi[:, 1], color="lightgreen")
        ax[i].set_xticks(())
        ax[i].set_yticks(())
    fig.legend(
        lib.plotting.create_legend_handles(("salmon", "lightgreen")),
        ["TagRFP", "EGFP"],
        loc="upper right",
        framealpha=1,
    )
    plt.tight_layout()
    return fig

@st.cache
def _get_data(min_len = None):
    """
    Loads all traces and converts them to a padded tensor
    """
    df = pd.DataFrame(pd.read_hdf("results/2_intensities/2_intensities.h5"))
    if min_len is not None:
        df = df.groupby(["file", "particle"]).filter(lambda x: len(x) > min_len)

    len_per_group = df.groupby(["file", "particle"]).apply(lambda x: len(x))
    max_len = np.max(len_per_group)
    n_groups = len(len_per_group)
    n_channels = 2

    X = np.zeros(shape=(n_groups, max_len, n_channels))
    for n, (_, group) in enumerate(df.groupby(["file", "particle"])):
        pad = max_len - len(group)
        X[n, pad:, 0] = group["int_c0"]
        X[n, pad:, 1] = group["int_c1"]
    return X


def _prepare_data(X, train_size=0.9):
    X_train, X_test = sklearn.model_selection.train_test_split(X, train_size=train_size, random_state = 1)
    X_train_len, X_test_len = len(X_train), len(X_test)

    X_train, X_test = [
        tf.data.Dataset.from_tensor_slices((tf.constant(Xi), tf.constant(Xo)))
        for (Xi, Xo) in ((X_train, X_train), (X_test, X_test))
    ]

    X_train, X_test = [
        data.shuffle(buffer_size=10 * BATCH_SIZE).batch(BATCH_SIZE) for data in (X_train, X_test)
    ]
    return (X_train, X_test), (X_train_len, X_test_len)


if __name__ == "__main__":
    X_raw = _get_data()
    X = lib.math.normalize_tensor(X_raw)
    print(X_raw.shape)
    MODEL_DIR = None
    N_TIMESTEPS = X.shape[1]
    N_FEATURES = X.shape[2]
    CALLBACK_TIMEOUT = 5
    EPOCHS = 10
    BATCH_SIZE = 32
    LATENT_DIM = 10

    # st.subheader("Raw data plots")
    # st.write(_plot_examples(X_raw))

    (X_train, X_test), (X_train_len, X_test_len) = _prepare_data(X)

    model, callbacks, initial_epoch = model_builder(
        model_dir=MODEL_DIR,
        model_build_f =lib.models.build_residual_conv_autoencoder,
        build_args=(N_FEATURES, LATENT_DIM, N_TIMESTEPS),
    )
    start = time.time()
    model.fit(
        x=X_train.repeat(),
        validation_data=X_test.repeat(),
        epochs=EPOCHS,
        steps_per_epoch=X_train_len // BATCH_SIZE,
        validation_steps=X_test_len // BATCH_SIZE,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    end = time.time()
    print("{:.2f}".format(end - start))