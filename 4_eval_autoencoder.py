import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.cluster
import sklearn.model_selection
import streamlit as st
import tensorflow.python.keras as keras

import lib.plotting
import lib.utils
import lib.math

np.random.seed(0)


def _plot_examples(X, nrows, ncols):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    ax = ax.ravel()
    rand_idx = np.random.randint(0, len(X), nrows * ncols).tolist()

    for i, r in enumerate(rand_idx):
        xi, = lib.utils.remove_zero_padding(X[r, ...])
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


def _plot_single_prediction(xi, xi_pred):
    fig, ax = plt.subplots()

    xi, xi_pred = lib.utils.remove_zero_padding(xi, xi_pred)
    ax.plot(xi[:, 0], color="salmon")
    ax.plot(xi[:, 1], color="lightgreen")
    ax.plot(xi_pred[:, 0], "--", color="darkred")
    ax.plot(xi_pred[:, 1], "--", color="darkgreen")

    ax.set_ylim(0, np.max((xi, xi_pred), axis=(0, 1, 2)))

    fig.legend(
        lib.plotting.create_legend_handles(("salmon", "lightgreen")),
        ["TagRFP", "EGFP"],
        loc="upper right",
        framealpha=1,
    )
    plt.tight_layout()
    return fig


@st.cache
def _get_data(include_steplength=False, min_len=1):
    """
    Loads all traces and converts them to a padded tensor
    """
    df = pd.DataFrame(pd.read_hdf("results/2_intensities/2_intensities.h5"))
    df = df.groupby(["file", "particle"]).filter(lambda x: len(x) > min_len)
    len_per_group = df.groupby(["file", "particle"]).apply(lambda x: len(x))
    max_len = np.max(len_per_group)
    n_groups = len(len_per_group)
    n_channels = 2

    columns = ["int_c0", "int_c1", "steplength"]
    if not include_steplength:
        columns.pop(-1)

    X_raw = np.zeros(shape=(n_groups, max_len, n_channels))
    for n, (_, group) in enumerate(df.groupby(["file", "particle"])):
        pad = max_len - len(group)
        X_raw[n, pad:, 0] = group["int_c0"]
        X_raw[n, pad:, 1] = group["int_c1"]

    X = lib.math.normalize_tensor(X_raw)
    return X, X_raw, len_per_group


def _get_cluster(encodings, n_clusters):
    clust = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)
    c_label = clust.fit_predict(encodings)

    pca = sklearn.decomposition.PCA(n_components=3)
    pca_z = pca.fit_transform(encodings)
    pca_z = np.column_stack((pca_z, c_label))
    pca_z = pd.DataFrame(pca_z, columns=["pc1", "pc2", "pc3", "label"])
    return pca_z, c_label


if __name__ == "__main__":
    MODEL_DIR = "logs/20191030-203317/model_010"
    X, X_raw, len_per_group = _get_data(min_len=40)
    # X = np.expand_dims(X[...,0], axis = 1) # only trained with c0

    model = keras.models.load_model(MODEL_DIR)
    encoder = model.layers[1]
    encodings = encoder.predict(X)

    n_clusters = st.slider("Number of clusters", 1, 5, 2)
    pca_z, c_label = _get_cluster(encodings, n_clusters)

    fig = px.scatter_3d(
        data_frame=pca_z,
        x="pc1",
        y="pc2",
        z="pc3",
        hover_name=pca_z.index.values,
        # hover_data = len_per_group.values,
        color=c_label,
        color_continuous_scale="viridis",
    )
    st.write(fig)

    selected_idx = int(st.text_input(label="Prediction ID to plot", value="0"))
    x_idx_true = X[selected_idx]
    x_idx_pred = np.squeeze(model.predict(np.expand_dims(X[selected_idx], axis=0)))
    fig = _plot_single_prediction(x_idx_true, x_idx_pred)
    st.write(fig)
