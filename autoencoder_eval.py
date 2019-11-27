import os
import re
import warnings

from tensorflow.keras.models import Model
import lib.models
from lib.math import mean_squared_error

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
    import tensorflow.python as tf
    from tensorflow.python import keras

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

from glob import glob

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.cluster
import sklearn.mixture
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.metrics
import streamlit as st
import lib.plotting
import lib.utils
import lib.math
from lib.utils import svg_write


def _get_encoding_layer(autoencoder: Model):
    encoder = Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer("encoded").output,
    )
    return encoder


def _get_npz(path):
    """
    Loads all traces
    """
    X = np.load(path, allow_pickle=True)["data"]
    return X


def _get_train_test_npz(path):
    """
    Loads train/test and normalization factor
    """
    f = np.load(path, allow_pickle=True)
    X_train, X_test, (mu, sg) = f["X_train"], f["X_test"], f["scale"]

    # Standardize
    X_train = np.array([(xi - mu) / sg for xi in X_train])
    X_test = np.array([(xi - mu) / sg for xi in X_test])
    return X_train, X_test, (mu, sg)


def _get_latest(MODEL_DIR, recency=1):
    """Fetches latest model in directory"""
    models = glob(MODEL_DIR + "/model*")
    try:
        latest = sorted(models)[-recency]
        return latest
    except IndexError:
        st.write("Index error. Does the directory actually contain models?")


def _get_features(X_true, model_path):
    """Predicts autoencoder features and saves them"""
    latest_model_path = _get_latest(model_path)
    autoencoder = keras.models.load_model(latest_model_path)
    encoder = _get_encoding_layer(autoencoder)
    if len(X_true.shape) == 3:
        # If trained with equal lengths
        features = encoder.predict(X_true)
        if len(features.shape) == 3:
            features = features[:, -1, :]
        X_pred = autoencoder.predict(X_true)
        mse = mean_squared_error(X_true, X_pred, axis=(1, 2))
    else:
        # If trained with variable lengths, need to do this sample-wise
        features, X_pred, mse = [], [], []
        for i in range(len(X_true)):
            xi = X_true[i]
            xi_true = np.expand_dims(xi, axis=0)
            xi_pred = autoencoder.predict_on_batch(xi_true).numpy()
            fi = encoder.predict_on_batch(xi_true)
            if len(fi.shape) == 3:
                fi = fi[0, -1, :]
            ei = mean_squared_error(xi_true, xi_pred, axis=(0, 1, 2))

            features.append(np.squeeze(fi))
            X_pred.append(np.squeeze(xi_pred))
            mse.append(np.squeeze(ei))

    mse = np.array(mse)
    features = np.array(features)
    X_pred = np.array(X_pred)

    if len(features.shape) == 3:
        features = features.reshape(features.shape[0], -1)

    return features, X_pred, mse


def _get_clusters(features, n_clusters, n_components):
    """Performs clustering and PCA for visualization"""
    pca = sklearn.decomposition.PCA(n_components=n_components)
    X_de = pca.fit_transform(features)
    explained_var = np.cumsum(np.round(pca.explained_variance_ratio_, 3))

    # cluster the decomposed
    clustering = sklearn.cluster.SpectralClustering(
        n_clusters=n_clusters, n_jobs=-1,
    )
    cluster_labels = clustering.fit_predict(X_de)

    # stack value decomposition and predicted labels
    X_de = np.column_stack((X_de, cluster_labels))
    X_de = pd.DataFrame(X_de)
    X_de["label"] = cluster_labels
    return X_de, cluster_labels, explained_var


if __name__ == "__main__":
    MODEL_DIR = "models/20191127-0722_multi_lstm_autoencoder_dim=128_variable_data=tracks-cme_split-c1_var.npz"
    # MODEL_DIR = "models/20191127-0016_multi_lstm_autoencoder_dim=32_variable_data=tracks-cme_split-c1_var.npz"
    # MODEL_DIR = sorted(glob("models/*"))[-1]
    st.write(MODEL_DIR)

    variable_len = True if re.search("variable", MODEL_DIR) else False

    # dataset used for fitting
    dataset = MODEL_DIR.split("data=")[-1]

    # real traces, for visual display
    X_vis_path = os.path.join("results/intensities", dataset[:-8] + "_var.npz")
    X_vis = _get_npz(X_vis_path)

    # fitted traces for autoencoder predictions
    X_data_path = os.path.join(
        "results/intensities", dataset[:-4] + "_traintest.npz"
    )

    X_train, X_test, (mu, sg) = _get_train_test_npz(X_data_path)

    X_true = X_test
    features, X_pred, mse = _get_features(X_true=X_true, model_path=MODEL_DIR)

    ST_RSEED = st.sidebar.number_input(
        min_value=0, max_value=999, value=0, label="Random seed"
    )

    ST_N_CLUSTERS = st.sidebar.slider(
        value=10, min_value=1, max_value=15, label="Number of clusters"
    )

    ST_N_COMPONENTS = st.sidebar.slider(
        value=features.shape[-1]//3,
        min_value=1,
        max_value=features.shape[-1],
        label="n components used for clustering",
    )

    ST_SHOW_PREDICTIONS = st.sidebar.radio(
        label="Show:", options=("predictions", "real-valued")
    )

    ST_ERROR_FILTER = st.sidebar.slider(
        max_value=np.max(mse),
        min_value=np.min(mse),
        value=np.max(mse),
        step=np.max(mse) / 100,
        format="%0.3f",
        label="Keep only traces with error below:",
    )

    np.random.seed(ST_RSEED)

    st.subheader("Total number of traces, N = {}".format(len(X_true)))

    mse_total = mse.copy()
    (filter_idx,) = np.where(mse < ST_ERROR_FILTER)
    X_true = X_true[filter_idx]
    X_pred = X_pred[filter_idx]
    features = features[filter_idx]
    mse = mse[filter_idx]

    st.subheader("Decomposition of features")
    pca_z, clusters, explained_var = _get_clusters(
        features=features,
        n_clusters=ST_N_CLUSTERS,
        n_components=ST_N_COMPONENTS,
    )

    fig = px.scatter_3d(
        pca_z.sample(frac=0.5, random_state=0), x=0, y=1, z=2, color="label"
    )
    st.write(fig)

    st.subheader("Reconstruction error")
    pca_z["mse"] = mse
    fig = px.scatter_3d(
        pca_z.sample(frac=0.5, random_state=0), x=0, y=1, z=2, color="mse"
    )
    st.write(fig)

    fig, ax = plt.subplots(ncols=2)
    ax[0].hist(
        mse_total[mse_total < np.quantile(mse_total, 0.97)],
        color="lightgrey",
        bins=30,
        edgecolor="darkgrey",
        density=True,
    )
    ax[0].axvline(ST_ERROR_FILTER, color="black", ls="--")
    ax[0].set_xlabel("MSE")

    ax[1].plot(np.arange(1, len(explained_var) + 1, 1), explained_var, "o-")
    ax[1].set_xlabel("n components")
    ax[1].set_ylabel("explained variance")
    ax[1].set_ylim(0, 1)
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    svg_write(fig)

    for n in range(ST_N_CLUSTERS):
        (selected_idx,) = np.where(clusters == n)
        if len(selected_idx) < len(X_true) // 100:
            st.subheader(
                "Cluster {} contains less than 1% of samples!".format(n)
            )
            continue
        st.subheader("Cluster {}".format(n))

        group_mse = mse[selected_idx]
        mean_group_mse = group_mse.mean()

        st.subheader(
            "Showing predictions for {} (N = {})".format(n, len(selected_idx))
        )
        # take only for the number of plots shown
        selected_mse = mse[selected_idx]
        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
        axes = axes.ravel()
        for n, ax in enumerate(axes):
            try:
                i = selected_idx[n]
                xi_t = X_true[i]
                xi_p = X_pred[i]
                xi_v = X_vis[i]

                xi_t, xi_p = lib.utils.remove_zero_padding(
                    xi_t, xi_p, padding="before"
                )

                if ST_SHOW_PREDICTIONS == "predictions":
                    lib.plotting.plot_c0_c1(
                        int_c0=xi_t[:, 0],
                        int_c1=xi_t[:, 1],
                        ax=ax,
                        alpha=0.5,
                        color0="cyan",
                        color1="orange",
                        separate_ax=False,
                    )

                    lib.plotting.plot_c0_c1(
                        int_c0=xi_p[:, 0],
                        int_c1=xi_p[:, 1],
                        ax=ax,
                        color0="blue",
                        color1="red",
                        separate_ax=False,
                    )

                elif ST_SHOW_PREDICTIONS == "real-valued":
                    lib.plotting.plot_c0_c1(
                        int_c0=xi_t[:, 0] * sg[0] + mu[0],
                        int_c1=xi_t[:, 1] * sg[1] + mu[1],
                        ax=ax,
                        alpha=0.5,
                        color0="cyan",
                        color1="orange",
                        separate_ax=False,
                    )

                    lib.plotting.plot_c0_c1(
                        int_c0=xi_p[:, 0] * sg[0] + mu[0],
                        int_c1=xi_p[:, 1] * sg[1] + mu[1],
                        ax=ax,
                        color0="blue",
                        color1="red",
                        separate_ax=False,
                    )
                else:
                    raise ValueError

                ax.set_title("E = {:.3f}\nl = {}".format(mse[i], len(xi_v)))
                ax.set_xticks(())
            except IndexError:
                fig.delaxes(ax)
        plt.tight_layout()
        svg_write(fig)
