from glob import glob
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.cluster
import sklearn.model_selection
import streamlit as st
from tensorflow.python import keras

import lib.math
import lib.plotting
import lib.utils
import sklearn.preprocessing

np.random.seed(0)


@st.cache
def _get_data(PATH):
    """
    Loads all traces
    """
    X_true = np.load(PATH)["data"]
    return X_true


def _get_latest(MODEL_DIR):
    """Fetches latest model in directory"""
    return sorted(glob(os.path.join(MODEL_DIR, "model*")))[-2]


def _get_clusters(X, n_clusters, n_components):
    """Performs clustering and PCA for visualization"""
    decomposer = sklearn.decomposition.PCA(n_components = n_components)
    X_de = decomposer.fit_transform(X)

    st.write(X_de.shape)

    # cluster the decomposed
    clustering = sklearn.cluster.SpectralClustering(n_clusters = n_clusters)
    cluster_labels = clustering.fit_predict(X_de)

    # stack value decomposition and predicted labels
    X_de = np.column_stack((X_de, cluster_labels))
    X_de = pd.DataFrame(X_de)
    X_de["label"] = cluster_labels
    return X_de, cluster_labels


if __name__ == "__main__":
    # remember the /
    MODEL_DIR = "models/20191106-184013DilatedConv/"
    PATH = "results/intensities/tracks-cme_split-c0c1_resampled-medianlen.npz"

    X_true = _get_data(PATH)
    X = lib.math.normalize_tensor(X_true[:, :, [0, 1]], feature_wise = False)

    model = keras.models.load_model(_get_latest(MODEL_DIR))
    encoder = model.layers[1]

    st.write("Model loaded: ", _get_latest(MODEL_DIR))

    features = encoder.predict(X)
    # Standard scale features, because the model output is not bounded
    features = sklearn.preprocessing.minmax_scale(features)

    st.subheader("Encoded features")
    st.write(features)

    n_clusters = st.sidebar.slider(
        value = 2, min_value = 2, max_value = 5, label = "n clusters"
    )
    n_components = st.sidebar.slider(
        value = 3,
        min_value = 3,
        max_value = features.shape[-1],
        label = "n components used for clustering",
    )
    pca_z, clusters = _get_clusters(
        features, n_clusters = n_clusters, n_components = n_components
    )
    st.write(pca_z.head())

    st.subheader("Decomposition of extracted extracted_features")
    fig = px.scatter_3d(pca_z.sample(frac = 0.5), x = 0, y = 1, z = 2, color = "label")
    st.write(fig)

    st.subheader("Average Clathrin/Auxilin trace. Number of traces, N = {}".format(len(X_true)))
    fig, ax = plt.subplots(nrows = n_clusters, figsize = (6, 10))
    for n in range(n_clusters):
        (selected_idx,) = np.where(clusters == n)
        mean_preds = np.mean(X_true[selected_idx], axis = 0)
        ax[n].set_title(
            "fraction = {:.2f}".format(len(selected_idx) / len(X_true))
        )
        lib.plotting.plot_c0_c1_errors(
            mean_int_c0 = mean_preds[:, 0], mean_int_c1 = mean_preds[:, 1], ax = ax[n], separate_ax = False
        )
    plt.tight_layout()
    st.write(fig)

    for n in range(n_clusters):
        (selected_idx,) = np.where(clusters == n)
        # take only for the number of plots shown
        selected_idx = selected_idx[0:16]

        st.subheader("Showing predictions for {}".format(n))
        fig, axes = plt.subplots(nrows = 4, ncols = 4)
        axes = axes.ravel()

        mean_preds = []
        for i, ax in zip(selected_idx, axes):
            xi_true = X[i]
            mean_preds.append(xi_true)
            lib.plotting.plot_c0_c1(
                int_c0 = xi_true[:, 0], int_c1 = xi_true[:, 1], ax = ax, separate_ax = False
            )
            ax.set_xticks(())
            ax.set_yticks(())
        plt.tight_layout()
        st.write(fig)