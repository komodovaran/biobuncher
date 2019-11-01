import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sklearn.cluster
import sklearn.decomposition
import sklearn.model_selection
import streamlit as st
from tensorflow import keras
from lib.models import *

from lib import utils

sns.set(context="paper", style="whitegrid", palette="pastel")


def z_score_norm(x):
    """
    Z-score normalizes array
    """
    return (x - np.mean(x)) / np.std(x)


@st.cache
def _get_data(length=50, n_each_class=200):
    """
    Make 3 types of sequence data with variable length
    """
    data = []
    labels = []
    for _ in range(n_each_class):

        y_noise = 0
        x_noisy = np.column_stack(
            (
                (
                    np.cos(np.linspace(0, 5, length))
                    + np.random.normal(0, 0.5, length)
                ),
                (
                    (
                        1
                        + np.sin(np.linspace(0, 5, length))
                        + np.random.normal(0, 0.5, length)
                    )
                ),
            )
        )

        y_wavy = 1
        x_wavy = np.column_stack(
            (
                (
                    np.cos(np.linspace(0, 20, length))
                    + np.random.normal(0, 0.5, length)
                ),
                (
                    (
                        1
                        + np.sin(np.linspace(0, 20, length))
                        + np.random.normal(0, 0.5, length)
                    )
                ),
            )
        )

        y_spikes = 2
        x_spikes = np.column_stack(
            (
                (
                    np.cos(np.linspace(0, 20, length))
                    + np.random.normal(0, 0.5, length)
                )
                ** 2,
                (
                    (
                        1
                        + np.sin(np.linspace(0, 20, length))
                        + np.random.normal(0, 0.5, length)
                    )
                    ** 2
                ),
            )
        )

        # Randomly cut the begining of traces and fill in with zeroes to mimick short traces
        zero = np.random.randint(1, length - 10)
        x_noisy[:zero] = 0
        x_wavy[:zero] = 0
        x_spikes[:zero] = 0

        x_noisy, x_wavy, x_spikes = [
            z_score_norm(x) for x in (x_noisy, x_wavy, x_spikes)
        ]

        labels.extend([y_noise, y_wavy, y_spikes])

        data.append(x_noisy)
        data.append(x_wavy)
        data.append(x_spikes)

    data = np.array(data)
    data = data.reshape((-1, length, 2))
    print("original shape ", data.shape)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data, labels, train_size=0.8
    )
    return X_train, X_test, y_train, y_test


def _get_predictions(X):
    encoder = model.layers[1]
    X_pred = model.predict(X)
    latents = encoder.predict(X)  # (n_samples, latent_dim)
    return X, X_pred, latents


def _get_predictions_vae(X):
    encoder = model.layers[1]
    X_pred = model.predict(X)
    z_mean, _, _ = encoder.predict(X)  # (n_samples, latent_dim)
    return X, X_pred, z_mean


def _get_cluster(n_clusters):
    clust = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)
    c_label = clust.fit_predict(latents)

    pca = sklearn.decomposition.PCA(n_components=3)
    pca_z = pca.fit_transform(latents)
    pca_z = np.column_stack((pca_z, c_label))
    pca_z = pd.DataFrame(pca_z, columns=["pc1", "pc2", "pc3", "label"])
    return pca_z, c_label


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = _get_data()
    N_TIMESTEPS = X_train.shape[1]
    N_FEATURES = X_train.shape[2]
    CALLBACK_TIMEOUT = 3
    EPOCHS = 100
    BATCH_SIZE = 64
    LATENT_DIM = 50

    st.subheader("Generated data shape:")
    st.write(X_train.shape)

    st.subheader("Raw data plots")
    fig, ax = plt.subplots(nrows=3, ncols=2)
    ax = ax.ravel()
    for i in range(6):
        xi, = utils.remove_zero_padding(X_train[i, ...])
        ax[i].plot(xi[:, 0], color="lightgreen", label="true c0")
        ax[i].plot(xi[:, 1], color="darkgreen", label="true c1")
        ax[i].legend(loc="upper right")
        ax[i].set_xticks(())
        ax[i].set_yticks(())
    st.write(fig)

    X_train_len, X_test_len = len(X_train), len(X_test)

    X_train, X_test = [
        tf.data.Dataset.from_tensor_slices((tf.constant(Xi), tf.constant(Xo)))
        for (Xi, Xo) in ((X_train, X_train), (X_test, X_test))
    ]

    X_train, X_test = [
        data.shuffle(buffer_size=10 * BATCH_SIZE).batch(BATCH_SIZE)
        for data in (X_train, X_test)
    ]

    try:
        model = tf.keras.models.load_model(
            "results/models/test_autoencoder.h5"
        )  # type: Model
    except OSError:
        model = create_vae(
            n_features=N_FEATURES,
            n_timesteps=N_TIMESTEPS,
            latent_dim=LATENT_DIM,
        )

        model.fit(
            x=X_train.repeat(),
            validation_data=X_test.repeat(),
            epochs=EPOCHS,
            steps_per_epoch=X_train_len // BATCH_SIZE,
            validation_steps=X_test_len // BATCH_SIZE,
            callbacks=[keras.callbacks.EarlyStopping(patience=30)],
        )

        model.save("results/models/test_autoencoder.h5")

    X_test = utils.batch_to_numpy(X_test)
    X_test, X_pred, latents = _get_predictions_vae(X_test)
    X_pred = X_pred.reshape(-1, N_TIMESTEPS, N_FEATURES)

    st.write("Test data shape: ", X_test.shape)

    n_clusters = st.sidebar.slider(
        min_value=1,
        max_value=6,
        value=3,
        label="Number of latent vector clusters:",
    )
    pca_z, c_label = _get_cluster(n_clusters)

    st.subheader("PCA of latent vectors")
    fig = px.scatter_3d(
        data_frame=pca_z,
        x="pc1",
        y="pc2",
        z="pc3",
        color=c_label,
        color_continuous_scale="viridis",
    )
    st.write(fig)

    selected_label = st.sidebar.selectbox(
        "Show predictions for label:", list(set(c_label))
    )
    selected_idx, = np.where(c_label == selected_label)
    selected_idx = selected_idx[0:9]  # take only for the number of plots shown

    st.subheader("Showing predictions for {}".format(selected_label))
    fig, axes = plt.subplots(nrows=3, ncols=3)
    axes = axes.ravel()

    n = 0
    for i, ax in zip(selected_idx, axes):
        xi_true, xi_pred = utils.remove_zero_padding(
            arr_true=X_test[i], arr_pred=X_pred[i], padding="before"
        )

        ax.plot(xi_true[:, 0], color="lightgreen", label="true c0")
        ax.plot(xi_true[:, 1], color="darkgreen", label="true c1")
        ax.plot(xi_pred[:, 0], color="salmon", label="pred c0")
        ax.plot(xi_pred[:, 1], color="darkred", label="pred c1")
        ax.plot([], [], label="length: {}".format(len(xi_true)))
        ax.legend(loc="upper right")
        ax.set_xticks(())
        ax.set_yticks(())

        n += 1

    plt.tight_layout()
    st.write(fig)
