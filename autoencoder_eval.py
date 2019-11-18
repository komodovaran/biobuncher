import os
import warnings

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
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.cluster
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.metrics
import streamlit as st

import lib.math
import lib.plotting
import lib.utils


@st.cache
def _get_npz(path, normalize, per_feature=False):
    """
    Loads all traces
    """
    X = np.load(path)["data"]
    if normalize:
        X = lib.math.normalize_tensor(X[:, :, [0, 1]], per_feature=per_feature)
    return X


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
    feature_path = os.path.join("results/extracted_features/", model_path[7:])
    st.write("Features will be saved to: ", feature_path)
    try:
        arrs = np.load(feature_path)
        features, X_pred, mse = arrs["features"], arrs["X_pred"], arrs["mse"]
    except FileNotFoundError:
        print("Predicting features")
        latest_model_path = _get_latest(model_path)
        model = keras.models.load_model(latest_model_path)
        encoder = model.layers[1]
        decoder = model.layers[2]
        features = encoder.predict(X_true)
        X_pred = decoder.predict(features)
        mse = sklearn.metrics.mean_squared_error(X_true, X_pred)
        np.savez(feature_path, features=features, X_pred=X_pred, mse=mse)
    return features, X_pred, mse


def _get_clusters(X, n_clusters, n_components):
    """Performs clustering and PCA for visualization"""
    decomposer = sklearn.decomposition.PCA(n_components=n_components)
    X_de = decomposer.fit_transform(X)

    st.write(X_de.shape)

    # cluster the decomposed
    clustering = sklearn.cluster.SpectralClustering(
        n_clusters=n_clusters, n_jobs=n_clusters
    )
    cluster_labels = clustering.fit_predict(X_de)

    # stack value decomposition and predicted labels
    X_de = np.column_stack((X_de, cluster_labels))
    X_de = pd.DataFrame(X_de)
    X_de["label"] = cluster_labels
    return X_de, cluster_labels


if __name__ == "__main__":
    # remember the /
    MODEL_DIR = "models/20191118-1651_residual_conv_autoencoder_dim=3__pr_ftr=False_data=tracks-cme_split-c1_res.npz"
    dataset = MODEL_DIR.split("data=")[-1]
    X_path = os.path.join("results/intensities", dataset)
    real_path = X_path  # [:-11] + "_res.npz" # 8 if no truncation, 11 if trunctaed 2 digit

    st.subheader(dataset)

    X = _get_npz(X_path, normalize=True)
    X_real = _get_npz(real_path, normalize=False)

    st.write("data path: ", X_path)
    st.write("real path: ", real_path)

    features, X_pred, mse = _get_features(X_true=X, model_path=MODEL_DIR)
    # Standard scale extracted_features, because the model output is not bounded
    features = sklearn.preprocessing.minmax_scale(features)

    st.subheader("Encoded extracted_features")
    st.write(features)

    n_clusters = st.sidebar.slider(
        value=2, min_value=2, max_value=15, label="n clusters"
    )
    n_components = st.sidebar.slider(
        value=3,
        min_value=1,
        max_value=features.shape[-1],
        label="n components used for clustering",
    )
    pca_z, clusters = _get_clusters(
        features, n_clusters=n_clusters, n_components=n_components
    )
    st.write(pca_z.head())

    st.subheader("Decomposition of extracted extracted_features")
    fig = px.scatter_3d(pca_z.sample(frac=0.5), x=0, y=1, z=2, color="label")
    st.write(fig)

    st.subheader(
        "Average Clathrin/Auxilin trace. Number of traces, N = {}".format(
            len(X)
        )
    )
    for n in range(n_clusters):
        fig, ax = plt.subplots()
        (selected_idx,) = np.where(clusters == n)
        mean_preds = np.mean(X_real[selected_idx], axis=0)
        ax.set_title(
            "N = {} (fraction = {:.2f})".format(
                len(selected_idx), len(selected_idx) / len(X)
            )
        )
        lib.plotting.plot_c0_c1_errors(
            mean_int_c0=mean_preds[:, 0],
            mean_int_c1=mean_preds[:, 1],
            ax=ax,
            separate_ax=False,
        )
        plt.tight_layout()
        st.write(fig)

    for n in range(n_clusters):
        (selected_idx,) = np.where(clusters == n)
        # take only for the number of plots shown
        selected_idx = selected_idx[0:16]

        st.subheader("Showing predictions for {}".format(n))
        fig, axes = plt.subplots(nrows=4, ncols=4)
        axes = axes.ravel()

        for i, ax in zip(selected_idx, axes):
            xi = X_real[i]
            lib.plotting.plot_c0_c1(
                int_c0=np.trim_zeros(xi[:, 0].ravel()),
                int_c1=np.trim_zeros(xi[:, 1].ravel()),
                ax=ax,
                separate_ax=False,
            )
            ax.set_xticks(())
            ax.set_yticks(())
        plt.tight_layout()
        st.write(fig)
