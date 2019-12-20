import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.cluster
import sklearn.decomposition
import sklearn.model_selection
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import *
import joypy as jp

sns.set(context = "paper", style = "darkgrid", palette = "pastel")


def _get_data(length = 50, n_each_class = 200):
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
                ((1 + np.sin(l(i(1, 20), 20, length)) + r(0, 0.2, length)) ** 3),
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
        data, train_size = 0.8
    )

    mu = np.mean(X_train, axis = (0, 1))
    sg = np.std(X_train, axis = (0, 1))
    X_train = (X_train - mu) / sg
    X_test = (X_test - mu) / sg

    return X_train, X_test


def multi_lstm_autoencoder(n_timesteps, n_features, latent_dim):
    inputs = Input(shape = (n_timesteps, n_features))

    x = Bidirectional(CuDNNLSTM(latent_dim), merge_mode = "mul", name = "encoded")(inputs)
    x = Activation("tanh")(x)
    x = RepeatVector(n_timesteps)(x)
    x = Bidirectional(CuDNNLSTM(latent_dim, return_sequences = True), merge_mode = "mul")(x)
    x = Activation("tanh")(x)

    outputs = TimeDistributed(Dense(n_features, activation = None))(x)

    autoencoder = Model(inputs = inputs, outputs = outputs)
    autoencoder.compile(optimizer = "adam", loss = "mse", metrics = ["mse"])
    return autoencoder


def _batch_to_numpy(tf_data, n_batches = -1):
    x = [xi.numpy() for xi, _ in tf_data.take(n_batches)]
    x = np.concatenate(x, axis = 0)
    return x


def _get_predictions(X, model):
    encoder = Model(
        inputs = model.input, outputs = model.get_layer("encoded").output
    )
    X_pred = model.predict(X)
    latents = encoder.predict(X)
    if len(latents.shape) == 3:
        latents = latents[:, -1, :]  # take final lstm output
    return X, X_pred, latents


def _get_cluster(n_clusters):
    clust = sklearn.cluster.AgglomerativeClustering(n_clusters = n_clusters)
    c_label = clust.fit_predict(latents)
    pca = sklearn.decomposition.PCA(n_components = 2)
    pca_z = pca.fit_transform(latents)
    pca_z = np.column_stack((pca_z, c_label))
    return pca_z, c_label


if __name__ == "__main__":
    X_train, X_test = _get_data()
    N_TIMESTEPS = X_train.shape[1]
    N_FEATURES = X_train.shape[2]
    CALLBACK_TIMEOUT = 3
    EPOCHS = 100
    BATCH_SIZE = 64
    LATENT_DIM = 128
    N_CLUSTERS = 3

    st.subheader("Generated data shape:")
    st.write(X_train.shape)

    X_train_len, X_test_len = len(X_train), len(X_test)

    X_train, X_test = [
        tf.data.Dataset.from_tensor_slices((tf.constant(Xi), tf.constant(Xo)))
        for (Xi, Xo) in ((X_train, X_train), (X_test, X_test))
    ]

    X_train, X_test = [
        data.shuffle(buffer_size = 10 * BATCH_SIZE).batch(BATCH_SIZE)
        for data in (X_train, X_test)
    ]

    model = multi_lstm_autoencoder(
        n_timesteps = N_TIMESTEPS, n_features = N_FEATURES, latent_dim = LATENT_DIM
    )

    model.fit(
        x = X_train.repeat(),
        validation_data = X_test.repeat(),
        epochs = EPOCHS,
        steps_per_epoch = X_train_len // BATCH_SIZE,
        validation_steps = X_test_len // BATCH_SIZE,
    )

    X_test = _batch_to_numpy(X_test)
    X_test, X_pred, latents = _get_predictions(X_test, model = model)
    X_pred = X_pred.reshape(-1, N_TIMESTEPS, N_FEATURES)

    joyfig = jp.joyplot(pd.DataFrame(latents), xlabels = True, ylabels = False, fade = True)

    st.write("Test data shape: ", X_test.shape)

    pca_z, c_label = _get_cluster(N_CLUSTERS)

    st.subheader("PCA of latent vectors")
    fig, ax = plt.subplots()
    ax.scatter(pca_z[:, 0], pca_z[:, 1], c = c_label, edgecolor = "black")

    for selected_label in range(3):
        fig, axes = plt.subplots(nrows = 5, ncols = 5)
        axes = axes.ravel()

        (selected_idx,) = np.where(c_label == selected_label)
        selected_idx = selected_idx[
                       0:25
                       ]  # take only for the number of plots shown
        st.subheader("Showing predictions for {}".format(selected_label))

        n = 0
        for n, ax in enumerate(axes):
            try:
                i = selected_idx[n]
                xi_true, xi_pred = X_test[i], X_pred[i]
                ax.plot(xi_true[:, 0], color = "lightgreen", label = "true c0", alpha = 0.3)
                ax.plot(xi_true[:, 1], color = "darkgreen", label = "true c1", alpha = 0.3)
                ax.plot(xi_pred[:, 0], color = "salmon", label = "pred c0")
                ax.plot(xi_pred[:, 1], color = "darkred", label = "pred c1")
                ax.plot([], [], label = "length: {}".format(len(xi_true)))
                ax.set_xticks(())
                ax.set_yticks(())
            except IndexError:
                fig.delaxes(ax)
        plt.tight_layout()
        plt.show()
