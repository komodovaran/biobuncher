import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture
import sklearn.model_selection
import sklearn.preprocessing
import streamlit as st
import tensorflow as tf
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.models import Model
from tensorflow.python import keras
from tqdm import tqdm
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

import lib.math
import lib.plotting
import lib.tfcustom
import lib.utils
from lib.tfcustom import VariableBatchGenerator
from lib.plotting import svg_write

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}


# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


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
    Loads train/test and normalization factor. Reverse normalization for plots only
    """
    f = np.load(path, allow_pickle=True)
    X_train, X_test, idx_train, idx_test, (mu, sg) = (
        f["X_train"],
        f["X_test"],
        f["idx_train"],
        f["idx_test"],
        f["scale"],
    )
    return X_train, X_test, idx_train, idx_test, (mu, sg)


def _get_latest(MODEL_DIR, recency=1):
    """Fetches latest model in directory"""
    models = glob(os.path.join(MODEL_DIR, "model*"))
    try:
        latest = sorted(models)[-recency]
        return latest
    except IndexError:
        st.write("Index error. Does the directory actually contain models?")


def _get_predictions(X, model_path):
    """
    Predicts autoencoder features and saves them. Saving as npz is required for interactivity,
    as predictions can take a couple of minutes
    """
    savename = os.path.split(model_path[:-4] + "_predictions")[-1] + ".npz"

    # See if there's a cached version
    try:
        f = np.load(
            "results/extracted_features/{}".format(savename), allow_pickle=True
        )
        X_true, X_pred, features, mse = (
            f["X_true"],
            f["X_pred"],
            f["features"],
            f["mse"],
        )
        return X_true, X_pred, features, mse

    # Else predict from scratch
    except FileNotFoundError:
        print("File not found. Trying to predict again if GPU is not busy!")
        latest_model_path = _get_latest(model_path)
        autoencoder = keras.models.load_model(
            latest_model_path, custom_objects={"gelu": lib.tfcustom.gelu}
        )
        encoder = _get_encoding_layer(autoencoder)

        # If trained with equal lengths
        if len(X.shape) == 3:
            X_true = X
            X_pred = autoencoder.predict(X_true)
            features = encoder.predict(X_true)
            mse = lib.math.mean_squared_error(X_true, X_pred, axis=(1, 2))

        # Batch as much as possible and predict on this
        else:
            X_ = VariableBatchGenerator(
                X=X.tolist(), max_batch_size=64, shuffle=True
            )
            X_ = tf.data.Dataset.from_generator(
                generator=X_,
                output_types=(tf.float64, tf.float64),
                output_shapes=((None, None, 2), (None, None, 2)),
            )

            # Sanity check
            for n, (Xi, _) in enumerate(X_):
                lib.plotting.sanity_plot(Xi.numpy(), "batch {}".format(n))
                if n == 3:
                    break

            # Predict on batches and unravel for single-item use
            X_true, X_pred, features, mse = [], [], [], []
            for n, (xi_true, _) in tqdm(enumerate(X_)):
                xi_pred = autoencoder.predict_on_batch(xi_true).numpy()
                xi_true = xi_true.numpy()

                X_true.extend(xi_true)
                X_pred.extend(xi_pred)

                fi = encoder.predict_on_batch(xi_true)
                ei = lib.math.mean_squared_error(xi_true, xi_pred, axis=(1, 2))

                features.extend(fi)
                mse.extend(ei)

            features = np.array(features)
            mse = np.array(mse)
            X_pred = np.array(X_pred)
            X_true = np.array(X_true)

        np.savez(
            "results/extracted_features/{}".format(savename),
            X_true=X_true,
            X_pred=X_pred,
            features=features,
            mse=mse,
        )
        return X_true, X_pred, features, mse


@st.cache
def _get_clusters(features, n_clusters, n_components):
    """Performs clustering and PCA for visualization"""
    # Decompose with PCA
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
    # MODEL_DIR = sorted(glob("models/*"))[-1]
    MODEL_DIR = "models/20191205-1320_single_lstm_autoencoder_dim=64_activation=elu_eps=0.1_zdim=2_variable_data=tracks-cme_var.npz"
    st.write(MODEL_DIR)

    INTENSITIES = "results/intensities"

    # dataset as arrays for fitting
    dataset_npz = MODEL_DIR.split("data=")[-1]

    # Actual data containing frame metadata
    dataset_df = dataset_npz[:-8] + ".h5"

    # df = pd.DataFrame(pd.read_hdf(dataset_df))

    # fitted traces for autoencoder predictions
    X_data_path = os.path.join(INTENSITIES, dataset_npz[:-4] + "_traintest.npz")
    X_train, X_test, idx_train, idx_test, (mu, sg) = _get_train_test_npz(
        X_data_path
    )

    X_true, X_pred, features, mse = _get_predictions(
        X=X_test, model_path=MODEL_DIR
    )

    features = np.array([f[-1, :] for f in features])
    X_true, X_pred, features = sklearn.utils.shuffle(X_true, X_pred, features)

    ST_RSEED = st.sidebar.number_input(
        min_value=0, max_value=999, value=0, label="Random seed"
    )

    ST_NCLUSTERS = st.sidebar.slider(
        value=2, min_value=1, max_value=15, label="Number of clusters"
    )

    ST_NCOMPONENTS = st.sidebar.slider(
        value=features.shape[-1] // 3,
        min_value=1,
        max_value=features.shape[-1],
        label="n components used for clustering",
    )

    ST_REALVALS = st.sidebar.checkbox(label="Display real values", value=True)

    ST_SEPARATEAX = st.sidebar.checkbox(label="Separate y-axis", value=False)

    ST_ERRORFILTER = st.sidebar.slider(
        max_value=np.max(mse),
        min_value=np.min(mse),
        value=np.max(mse),
        step=np.max(mse) / 100,
        format="%0.3f",
        label="Keep only traces with error below:",
    )

    np.random.seed(ST_RSEED)

    st.subheader(
        "Total number of traces in test data, N = {}".format(len(X_true))
    )

    mse_total = mse.copy()
    (filter_idx,) = np.where(mse < ST_ERRORFILTER)
    X_true = X_true[filter_idx]
    X_pred = X_pred[filter_idx]
    features = features[filter_idx]
    mse = mse[filter_idx]

    st.subheader("Decomposition of features")
    pca_z, clusters, explained_var = _get_clusters(
        features=features, n_clusters=ST_NCLUSTERS, n_components=ST_NCOMPONENTS,
    )

    fig = px.scatter_3d(
        pca_z.sample(frac=0.5, random_state=0), x=0, y=1, z=2, color="label"
    )
    st.write(fig)

    fig, ax = plt.subplots(ncols=2)
    ax[0].hist(
        mse_total[mse_total < np.quantile(mse_total, 0.90)],
        color="lightgrey",
        bins=30,
        edgecolor="darkgrey",
        density=True,
    )
    ax[0].axvline(ST_ERRORFILTER, color="black", ls="--")
    ax[0].set_xlabel("MSE")

    ax[1].plot(np.arange(1, len(explained_var) + 1, 1), explained_var, "o-")
    ax[1].set_xlabel("n components")
    ax[1].set_ylabel("explained variance")
    ax[1].set_ylim(0, 1)
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    svg_write(fig)

    for i in range(ST_NCLUSTERS):
        (selected_idx,) = np.where(clusters == i)
        if len(selected_idx) < len(X_true) // 100:
            st.subheader(
                "Cluster {} contains less than 1% of samples!".format(i)
            )
            continue
        st.subheader("Cluster {}".format(i))

        group_mse = mse[selected_idx]
        mean_group_mse = group_mse.mean()

        st.subheader(
            "Showing predictions for {} (N = {})".format(i, len(selected_idx))
        )

        selected_mse = mse[selected_idx]
        fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(10, 10))
        axes = axes.ravel()
        for j, ax in enumerate(axes):
            try:
                idx = selected_idx[j]
                xi_t = X_true[idx]
                xi_p = X_pred[idx]

                for k, xk in enumerate((xi_t, xi_p)):
                    if ST_REALVALS:
                        c0 = xk[:, 0] * sg[0] + mu[0]
                        c1 = xk[:, 1] * sg[1] + mu[1]
                    else:
                        c0 = xk[:, 0]
                        c1 = xk[:, 1]

                    inner_axes = lib.plotting.plot_c0_c1(
                        int_c0=c0,
                        int_c1=c1,
                        ax=ax,
                        alpha=0.5,
                        color0="orange" if k == 0 else "red",
                        color1="cyan" if k == 0 else "blue",
                        separate_ax=ST_SEPARATEAX,
                    )

                    if k == 0 and ST_SEPARATEAX:
                        inner_axes[-1].set_yticks(())

                ax.set_title("E = {:.3f}\nl = {}".format(mse[idx], len(xi_t)))
                ax.set_xticks(())
            except IndexError:
                # Delete extra axes if cluster too small
                fig.delaxes(ax)
        plt.tight_layout()
        svg_write(fig)
