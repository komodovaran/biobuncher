import os
import re
import warnings
from tensorflow.keras.models import Model

from lib.math import mean_squared_error

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = FutureWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
    import tensorflow.python as tf
    from tensorflow.python import keras

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.cluster
import sklearn.mixture
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.metrics
import streamlit as st
import lib.math
import lib.plotting
import lib.utils


def _get_npz(path, normalize = False, per_feature = False):
    """
    Loads all traces
    """
    X = np.load(path, allow_pickle = True)["data"]
    return X

def _get_train_test_npz(path, normalize):
    """
    Loads train/test and normalization factor
    """
    f = np.load(path, allow_pickle = True)
    X_train, X_test = f["X_train"], f["X_test"]
    if normalize == "sample":
        X_train, X_test = [
            lib.math.maxabs_tensor(X, per_feature = False)
            for X in (X_train, X_test)
        ]
    elif normalize == "dataset":
        X_train_max = f["scale"]
        X_train, X_test = [np.divide(X, X_train_max) for X in (X_train, X_test)]
    else:
        raise ValueError
    return X_train, X_test



def _get_latest(MODEL_DIR, recency = 1):
    """Fetches latest model in directory"""
    models = glob(MODEL_DIR + "/model*")
    try:
        latest = sorted(models)[-recency]
        return latest
    except IndexError:
        st.write("Index error. Does the directory actually contain models?")


def _get_features(X_true, model_path):
    """Predicts autoencoder features and saves them"""
    feature_path = os.path.join("../results/extracted_features/", model_path[7:])
    # try:
    #     arrs = np.load(feature_path)
    #     features, X_pred, mse = arrs["features"], arrs["X_pred"], arrs["mse"]
    # except FileNotFoundError:
    latest_model_path = _get_latest(model_path)
    autoencoder = keras.models.load_model(latest_model_path)
    encoder = Model(inputs = autoencoder.input,
                    outputs = autoencoder.get_layer("encoded").output)

    features = encoder.predict(X_true)

    X_pred = []
    for i in range(len(X_true)):
        xi = tf.expand_dims(X_true[i], axis = 0)
        X_pred.append(autoencoder.predict(xi))

    mse = mean_squared_error(X_true, X_pred, axis = (1, 2))

    np.savez(feature_path, features = features, X_pred = X_pred, mse = mse)
    return features, X_pred, mse


def _get_clusters(features, n_clusters, n_components):
    """Performs clustering and PCA for visualization"""
    decomposer = sklearn.decomposition.PCA(
        n_components = n_components,
    )
    X_de = decomposer.fit_transform(features)
    # X_de = features

    # cluster the decomposed
    clustering = sklearn.cluster.SpectralClustering(
        n_clusters = n_clusters, n_jobs = -1,
    )
    cluster_labels = clustering.fit_predict(X_de)

    # stack value decomposition and predicted labels
    X_de = np.column_stack((X_de, cluster_labels))
    X_de = pd.DataFrame(X_de)
    X_de["label"] = cluster_labels
    return X_de, cluster_labels


if __name__ == "__main__":
    MODEL_DIR = "models/20191126-1301_lstm_autoencoder_dim=50_norm=dataset_bidir=True_prftr=False_data=tracks-cme_split-c1_res.npz"

    pr_ftr_norm = True if re.search("pr_ftr=True", MODEL_DIR) else False
    dataset_norm = True if re.search("norm=dataset", MODEL_DIR) else False

    # dataset used for fitting
    dataset = MODEL_DIR.split("data=")[-1]

    # real traces, for visual display
    X_vis_path = os.path.join("../results/intensities", dataset[:-8] + "_res.npz")
    X_vis = _get_npz(X_vis_path, normalize = False)

    # fitted traces for autoencoder predictions
    X_true_path = os.path.join("../results/intensities", dataset)
    if not dataset_norm:
        X_true = _get_npz(X_true_path, normalize = True, per_feature = pr_ftr_norm)
    else:
        X_true = _get_npz(X_true_path, normalize = False)
        X_true = lib.math.div0(X_true, np.max(X_true, axis = (0, 1, 2)))


    st.subheader(dataset)
    st.subheader(X_true_path)
    st.subheader(X_true.shape)
    st.subheader(X_vis_path)
    st.subheader(X_vis.shape)

    features, X_pred, mse = _get_features(X_true = X_true, model_path = MODEL_DIR)
    features = sklearn.preprocessing.minmax_scale(features)

    ST_RSEED = st.sidebar.number_input(
        min_value = 0, max_value = 999, value = 0, label = "Random seed"
    )

    ST_N_CLUSTERS = st.sidebar.slider(
        value = 1, min_value = 1, max_value = 15, label = "Number of clusters"
    )

    ST_N_COMPONENTS = st.sidebar.slider(
        value = 3,
        min_value = 1,
        max_value = features.shape[-1],
        label = "n components used for clustering",
    )

    ST_SHOW_TRUE = st.sidebar.checkbox(value = True, label = "Show true traces")
    ST_SHOW_PRED = st.sidebar.checkbox(
        value = True, label = "Show predicted traces"
    )
    ST_SHOW_VIS = st.sidebar.checkbox(
        value = True, label = "Show original traces"
    )

    ST_ERROR_FILTER = st.sidebar.slider(
        max_value = np.max(mse),
        min_value = np.min(mse),
        value = np.max(mse),
        step = np.max(mse) / 100,
        format = "%0.3f",
        label = "Keep only traces with error below:",
    )

    np.random.seed(ST_RSEED)

    st.subheader("Total number of traces, N = {}".format(len(X_true)))

    mse_total = mse.copy()
    (filter_idx,) = np.where(mse < ST_ERROR_FILTER)
    X_true = X_true[filter_idx]
    X_vis = X_vis[filter_idx]
    X_pred = X_pred[filter_idx]
    features = features[filter_idx]
    mse = mse[filter_idx]

    st.subheader("Decomposition of features")
    pca_z, clusters = _get_clusters(
        features = features,
        n_clusters = ST_N_CLUSTERS,
        n_components = ST_N_COMPONENTS,
    )
    fig = px.scatter_3d(pca_z.sample(frac = 0.5, random_state = 0), x = 0, y = 1, z = 2, color = "label")
    st.write(fig)

    st.subheader("Reconstruction error")
    pca_z["mse"] = mse
    fig = px.scatter_3d(pca_z.sample(frac = 0.5, random_state = 0), x = 0, y = 1, z = 2, color = "mse")
    st.write(fig)

    st.subheader("Error distribution for all samples")
    fig, ax = plt.subplots()
    ax.hist(mse_total[mse_total < np.quantile(mse_total, 0.98)], color = "lightgrey", bins = 40, edgecolor = "darkgrey",
            density = True)
    ax.axvline(ST_ERROR_FILTER, color = "black", ls = "--")
    st.write(fig)

    for n in range(ST_N_CLUSTERS):

        (selected_idx,) = np.where(clusters == n)
        if len(selected_idx) < len(X_vis) // 100:
            st.subheader("Cluster {} contains less than 1% of samples!".format(n))
            continue
        st.subheader("Cluster {}".format(n))

        group_X_vis = np.mean(X_vis[selected_idx], axis = 0)
        group_X_out = np.mean(X_pred[selected_idx], axis = 0)
        group_mse = mse[selected_idx]
        mean_group_mse = group_mse.mean()

        fig, ax = plt.subplots()
        ax.set_title(
            "N = {} (fraction = {:.2f})\nmean squared error = {:.2e}".format(
                len(selected_idx),
                len(selected_idx) / len(X_true),
                mean_group_mse,
            )
        )
        # ax.plot(X_vis[selected_idx, :, 0].T, color = "red", alpha = 0.1)
        # ax.plot(X_vis[selected_idx, :, 1].T, color = "green", alpha = 0.1)
        lib.plotting.plot_c0_c1_errors(
            mean_int_c0 = group_X_vis[:, 0],
            mean_int_c1 = group_X_vis[:, 1],
            ax = ax,
            separate_ax = False,
        )

        plt.tight_layout()
        st.write(fig)

        st.subheader(
            "Showing predictions for {} (N = {})".format(n, len(selected_idx))
        )

        # take only for the number of plots shown
        selected_mse = mse[selected_idx]
        selected_idx = selected_idx[0:25]
        fig, axes = plt.subplots(nrows = 5, ncols = 5)
        axes = axes.ravel()
        for i, ax in zip(selected_idx, axes):
            xi_t = X_true[i]
            xi_p = X_pred[i]
            xi_v = X_vis[i]

            if ST_SHOW_VIS:
                lib.plotting.plot_c0_c1(
                    int_c0 = xi_v[:, 0],
                    int_c1 = xi_v[:, 1],
                    ax = ax,
                    separate_ax = False,
                )

            if ST_SHOW_TRUE:
                lib.plotting.plot_c0_c1(
                    int_c0 = xi_t[:, 0],
                    int_c1 = xi_t[:, 1],
                    ax = ax,
                    alpha = 0.9,
                    separate_ax = False,
                )

            if ST_SHOW_PRED:
                lib.plotting.plot_c0_c1(
                    int_c0 = xi_p[:, 0],
                    int_c1 = xi_p[:, 1],
                    ax = ax,
                    color0 = "darkred",
                    color1 = "darkgreen",
                    separate_ax = False,
                )

            ax.set_title("E = {:.3f}".format(mse[i]))
            ax.set_xticks(())
        plt.tight_layout()
        st.write(fig)
