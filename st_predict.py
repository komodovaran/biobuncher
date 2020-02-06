import os
import re
from glob import glob
from typing import Iterable

import matplotlib.pyplot as plt

# noinspection PyUnresolvedReferences
import mpl_scatter_density
import numpy as np
import pandas as pd
import parmap
import seaborn as sns
import sklearn.manifold
import sklearn.metrics
import sklearn.mixture
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.utils
import sklearn.utils.random
import streamlit as st
import tensorflow as tf
import umap.umap_ as umap
from matplotlib.ticker import (
    MaxNLocator,
    AutoMinorLocator,
    FixedFormatter,
    NullFormatter,
)
from scipy.cluster import hierarchy
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Model
from tensorflow.python import keras
from tqdm import tqdm
from hdbscan import HDBSCAN
import lib.math

import lib.globals
import lib.math
import lib.models
import lib.plotting
import lib.utils
from lib.plotting import svg_write
from lib.tfcustom import VariableTimeseriesBatchGenerator, gelu
from lib.utils import get_index, timeit

sns.set_style("dark")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}


@timeit
@st.cache
def _load_npz(path):
    """
    Loads all traces

    Args:
        path (str)
    """
    return np.load(path, allow_pickle=True)["data"]


@timeit
@st.cache
def _load_df(path, index=None):
    """
    Loads original dataframe that matches array
    Args:
        path (str)
        index (np.ndarray)
    """
    df = pd.DataFrame(pd.read_hdf(path, key="df"))
    if index is not None:
        df = df[df["id"].isin(index)]
    return df


@timeit
@st.cache
def _load_train_test_info(path):
    """
    Load indices for train/test split on the given dataset from a model dir.

    Args:
        path (str)
    """
    f = np.load(os.path.join(path, "info.npz"), allow_pickle=True)
    idx_train, idx_test, mu, sg = f["info"]
    return idx_train, idx_test, mu, sg


def _find_default_data_st_selection(model_path, data_dir):
    """
    Finds the default dataset name and streamlit selection index,
    by comparing dataset in model name (data=...) with dataset directory.

    For faulty names, returns None, which can be used for Return in the main()
    function

    Args:
        model_path (str)
        data_dir (list)
    """
    name = re.search("data=.*.npz", model_path)
    try:
        name = name[0].split("=")[1]
    except TypeError:
        return None, None

    idx = None
    for i, d in enumerate(data_dir):
        if os.path.basename(d) == name:
            idx = i

    return name, idx


@st.cache
def _standardize_train_test(X, mu, sigma, idx_train=None, idx_test=None):
    """
    Uses the indices to find the right train/test samples, and normalizes them.

    Args:
        X (np.ndarray)
        mu (float)
        sigma (float)
        idx_train (np.ndarray or None)
        idx_test (np.ndarray or None)
    """
    if idx_train is None or idx_test is None:
        X = lib.math.standardize(X, mu, sigma)
    else:
        X_train, X_test = (
            X[idx_train],
            X[idx_test],
        )  # type: np.ndarray, np.ndarray
        X = [lib.math.standardize(X, mu, sigma) for X in (X_train, X_test)]
    return X


@timeit
def _prepare_data(X, mu_train, sigma_train):
    """
    Args:
        X (np.ndarray)
        mu_train (float)
        sigma_train (float)
    """
    X_true = lib.math.standardize(X=X, mu=mu_train, sigma=sigma_train)
    idx_true = np.arange(len(X_true))
    return X_true, idx_true, mu_train, sigma_train


def _latest_model(model_dir, recency=1):
    """
    Fetches latest model path in directory.

    Args:
        model_dir (str)
        recency (int)
    """
    models = glob(os.path.join(model_dir, "model*"))
    try:
        latest = sorted(models)[-recency]
        return latest
    except IndexError:
        st.write("Index error. Does the directory actually contain models?")


def _get_encoding_layer(
    autoencoder, encoding_layer_names=("encoded", "z_sample", "z_mu"),
):
    """
    Gets the encoding layer from a model.

    Args:
        autoencoder (Model)
        encoding_layer_names (Iterable[str])
    """
    encoder = None
    for name in encoding_layer_names:
        try:
            encoder = Model(
                inputs=autoencoder.input,
                outputs=autoencoder.get_layer(name).output,
            )
        except ValueError:
            continue
    if encoder is None:
        raise ValueError("No matching encoding layers found")
    return encoder


def _predict(X_true, idx_true, model_path, savename, single_feature):
    """
    Predicts autoencoder features and saves them. Saving as npz is required for
    interactivity, as predictions use the GPU (which may not be available while
    training). Warning: order is not preserved due to batching!

    Args:
        X_true (np.ndarray)
        idx_true (np.ndarray)
        model_path (str)
        savename (str)
        single_feature (int or None)
    """
    # See if there's a cached version
    try:
        f = np.load(
            os.path.join(lib.globals.encodings_dir, savename), allow_pickle=True
        )
        X_true, X_pred, features, mse, indices = (
            f["X_true"],
            f["X_pred"],
            f["features"],
            f["mse"],
            f["indices"],
        )
    # Else predict from scratch
    except FileNotFoundError:
        print("No predictions found. Re-predicting.")
        latest_model_path = _latest_model(model_path)

        autoencoder = keras.models.load_model(
            latest_model_path,
            custom_objects={"gelu": gelu, "f1_m": lib.math.f1_m},
        )

        encoder = _get_encoding_layer(autoencoder)

        # Batch as much as possible to speed up prediction
        if len(X_true[0].shape) == 2:
            n_features = X_true[0].shape[-1]
        else:
            raise ValueError(
                "Trouble guessing number of features from data."
                "Reshape it to (-1, 1) if data is single feature"
            )

        X_ = VariableTimeseriesBatchGenerator(
            X=X_true.tolist(),
            indices=idx_true,
            max_batch_size=512,
            shuffle_samples=True,
            shuffle_batches=True,
        )

        indices = X_.indices

        X_ = tf.data.Dataset.from_generator(
            generator=X_,
            output_types=(tf.float64, tf.float64),
            output_shapes=((None, None, n_features), (None, None, n_features),),
        )

        # Predict on batches and unravel for single-item use
        X_true, X_pred, features, mse = [], [], [], []
        for xi_true, _ in tqdm(X_):
            # Predict encoding
            if single_feature is not None:
                _xi_true = np.expand_dims(xi_true[..., single_feature], axis=-1)
            else:
                _xi_true = xi_true
            fi = encoder.predict_on_batch(_xi_true)

            # Predict reconstruction
            try:
                xi_pred = autoencoder.predict_on_batch(_xi_true)

                # Calculate error of reconstruction
                ei = lib.math.mean_squared_error(
                    xi_true.numpy(), xi_pred.numpy(), axis=(1, 2)
                )
            except ValueError:
                xi_pred = _xi_true
                ei = np.zeros(xi_true.shape[0])

            # Make sure they're numpy arrays now and not EagerTensors!
            X_true.extend(np.array(xi_true))
            X_pred.extend(np.array(xi_pred))
            features.extend(np.array(fi))
            mse.extend(np.array(ei))

        X_true, X_pred, features, mse = map(
            np.array, (X_true, X_pred, features, mse)
        )

    np.savez(
        os.path.join(lib.globals.encodings_dir, savename),
        X_true=X_true,
        X_pred=X_pred,
        features=features,
        mse=mse,
        indices=indices,
    )

    if not st._is_running_with_streamlit:
        print(
            "Predictions done. You can stop the script now and run streamlit to visualize."
        )

    return X_true, X_pred, features, mse, indices


@st.cache
def _pca(features, embed_into_n_components=None):
    """
    Calculates the PCA of raw input features.

    Args:
        features (np.ndarray)
        embed_into_n_components (int or None)
    """
    if embed_into_n_components is None:
        embed_into_n_components = features.shape[-1]
    pca = sklearn.decomposition.PCA(n_components=embed_into_n_components)
    pca_features = pca.fit_transform(features)
    explained_var = np.cumsum(np.round(pca.explained_variance_ratio_, 3))
    return pca_features, explained_var


@st.cache
def _umap_embedding(features, embed_into_n_components, savename):
    """
    Calculates the UMAP embedding of raw input features for visualization.

    Args:
        features (np.ndarray)
        embed_into_n_components (int)
    """
    savename = os.path.join(lib.globals.umap_dir, savename)

    try:
        u = np.load(savename, allow_pickle=True)["umap"]
    except FileNotFoundError:
        u = umap.UMAP(
            n_components=embed_into_n_components,
            random_state=42,
            n_neighbors=100,
            min_dist=0.0,
            init="spectral",
            verbose=True,
        )
        u = u.fit_transform(features)
        np.savez(savename, umap=u)
    return u


@timeit
@st.cache
def _cluster_kmeans(features, n_clusters):
    """
    Cluster points using K-means

    Args:
        features (np.ndarray)
        n_clusters (int)
    """
    clf = MiniBatchKMeans(n_clusters=n_clusters)
    labels = clf.fit_predict(features)
    centers = clf.cluster_centers_
    return labels, centers


@timeit
@st.cache
def _cluster_gmm(features, n_clusters):
    """
    Cluster points using Gaussian Mixture Model

    Args:
        features (np.ndarray)
        n_clusters (int)
    """
    clf = GaussianMixture(n_components=n_clusters, covariance_type="full")
    labels = clf.fit_predict(features)
    centers = clf.means_
    return labels, centers


@timeit
@st.cache
def _cluster_hdbscan(features, min_cluster_size, merge_limit):
    """
    Cluster points using HDBSCAN
    """
    clf = HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=merge_limit,
    )
    clf.fit_predict(features)
    labels = clf.labels_
    centers = lib.math.cluster_centers(features, labels)
    return labels, centers


@timeit
@st.cache
def _calculate_kde(x, y):
    """
    Calculates KDE

    Args:
        x (np.ndarray)
        y (np.ndarray)
    """
    X, Y, Z = lib.math.kde_2d(x, y)
    return X, Y, Z


def _resample_traces(traces, length, normalize):
    """
    Resamples all input traces to a given length and normalizes them for easier overlay.

    Args:
        traces (np.ndarray)
        length (int)
        normalize (bool)
    """
    traces_re = np.array(
        parmap.map(
            lib.math.resample_timeseries, traces, length, pm_processes=16
        )
    )
    if normalize:
        new = []
        for trace in traces_re:
            t = trace / trace.max(axis=(0, 1))
            t -= trace.min(axis=(0, 1))
            new.append(t)

        traces_re = np.array(new)
    return traces_re


def _save_cluster_sample_indices(cluster_sample_indices, savename):
    """
    Save sample indices for clusters to later retrieve from dataset

    Args:
        cluster_sample_indices (pd.DataFrame)
        savename (str)
    """
    cluster_sample_indices.to_hdf(
        os.path.join(lib.globals.cluster_idx_dir, savename), key="df"
    )


def _find_peaks(X, n_frames=3, n_std=2):
    """
    Finds peaks for input list of arrays.

    Args:
        X (np.ndarray)
        n_frames (int)
        n_std (int or float)
    """
    has_peak = 0
    has_no_peak = 0

    for xi in X:
        mid = np.mean(xi[:, 1])
        dev = np.std(xi[:, 1])
        cutoff = mid + n_std * dev
        if np.sum(xi[:, 1] > cutoff) >= n_frames:
            has_peak += 1
        else:
            has_no_peak += 1
    return has_peak, has_no_peak


def _clath_aux_peakfinder(X):
    """
    Clathrin/auxilin specific block for doing a weak check for peaks.
    Not very tweaked and only used as a weak guidance.

    Args:
        X (np.ndarray)
    """

    has_peak, has_no_peak = _find_peaks(X)
    st.write("{} traces with peaks".format(has_peak))
    st.write("{} traces with no peaks".format(has_no_peak))
    st.write(
        "{:.1f} % of traces have peaks".format(
            has_peak / (has_peak + has_no_peak) * 100
        )
    )


def _plot_explained_variance(explained_variance):
    """
    Plots explained PCA variance.

    Args:
        explained_variance (np.ndarray)
    """
    fig, ax = plt.subplots()
    ax.plot(
        np.arange(1, len(explained_variance) + 1, 1), explained_variance, "o-"
    )
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Explained variance")
    ax.set_ylim(0, 1)
    ax.axhline(0.9, color="black", ls=":")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    svg_write(fig)


@timeit
def _plot_scatter(
    features,
    cluster_labels=None,
    subsample=False,
    density=False,
    scalebar=False,
):
    """
    Plots lower dimensional embedding with predicted cluster labels.

    Args:
        features (np.ndarray)
        cluster_labels (np.ndarray or None)
    """
    f, cl = features, cluster_labels
    if subsample:
        f, cl = sklearn.utils.resample(
            f, cl, n_samples=subsample, replace=False
        )
    f0, f1 = f[:, 0], f[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")

    if cluster_labels is None:
        c = ax.scatter_density(f0, f1, cmap="viridis", dpi=20)
        if scalebar:
            fig.colorbar(c, label="Datapoints per pixel")
    else:
        # Make discrete colormap
        lmin = min(cl[cl != -1])
        lmax = max(cl[cl != -1])
        label_cmap = plt.get_cmap("viridis", lmax - lmin + 1)

        # Plot clusters and labels
        ax.scatter_density(
            f0[cl != -1],
            f1[cl != -1],
            c=cl[cl != -1],
            cmap=label_cmap,
            dpi=20,
            alpha=0.75,
        )

        # Try to plot outliers if supported by clustering methodcalculated
        # else ignore
        try:
            ax.scatter_density(
                f0[cl == -1], f1[cl == -1], color="grey", alpha=0.75, dpi=20
            )
        except ValueError:
            pass

        # Plot cluster centers
        for i in range(len(set(cl))):
            fi = f[cl == i]
            m = np.mean(fi, axis=0)
            ax.annotate(
                xy=m,
                s=i,
                bbox=dict(boxstyle="square", fc="w", ec="grey", alpha=0.9),
            )

    # Plot density contour overlay
    if density:
        xx, yy, zz = _calculate_kde(f0, f1)
        ax.contour(xx, yy, zz, cmap="Greys", alpha=0.75)

    ax.set_xlabel("C0")
    ax.set_ylabel("C1")
    plt.tight_layout()
    svg_write(fig)


def _plot_mse(
    original_mse, mse_filter,
):
    """
    Plots adjustables (MSE and confidence thresholds), given the original MSE
    and confidences (before filtering applied).

    Args:
        original_mse (np.ndarray)
        mse_filter (np.ndarray)
    """
    fig, ax = plt.subplots()
    ax.hist(
        original_mse,
        color="lightgrey",
        bins=np.arange(0, max(original_mse), 0.01),
        edgecolor="darkgrey",
        density=True,
    )
    ax.axvline(mse_filter, color="black", ls="--")
    ax.set_xlim(0, np.quantile(original_mse, 0.95))
    ax.set_xlabel("MSE")
    ax.set_ylabel("Probability density")

    plt.tight_layout()
    svg_write(fig)


def _plot_traces_preview(
    X_true,
    X_pred,
    mse,
    sample_indices,
    plot_real_values,
    separate_y_ax,
    nrows,
    ncols,
    mu,
    sg,
    colors,
):
    """
    Plots a subset of clustered traces for inspection

    Args:
        X_true (np.ndarray)
        X_pred (np.ndarray)
        mse (np.ndarray)
        sample_indices (np.ndarray)
        plot_real_values (bool)
        separate_y_ax (bool)
        nrows (int)
        ncols (int)
        mu (float or np.ndarray)
        sg (float or np.ndarray)
        colors (list of str)
    """
    X_true, X_pred, mse, sample_indices = sklearn.utils.shuffle(
        X_true, X_pred, mse, sample_indices
    )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        try:
            xi_true = X_true[i]
            xi_pred = X_pred[i]
            ei = mse[i]
            idx = sample_indices[i]
        except IndexError:
            fig.delaxes(ax)
            continue

        rng = range(len(xi_true))

        ax.set_title(
            "E = {:.2f}, L = {}\n" "idx = {}\n".format(ei, len(xi_true), idx),
            fontsize=8,
        )
        ax.set_xticks(())

        if plot_real_values:
            xi_true = xi_true * sg + mu
            xi_pred = xi_pred * sg + mu

        for c in range(xi_true.shape[-1]):
            if not separate_y_ax:
                ax.plot(rng, xi_true[:, c], alpha=0.5, color=colors[c])
                try:
                    ax.plot(rng, xi_pred[:, c], color=colors[c])
                except IndexError:
                    pass
            else:
                ax_ = ax.twinx()
                ax_.plot(rng, xi_true[:, c], alpha=0.5, color=colors[c])
                try:
                    ax_.plot(rng, xi_pred[:, c], color=colors[c])
                except IndexError:
                    pass

                ax_.set_yticks(())
                ax.set_yticks(())

        # Works only for 2C clathrin/auxilin traces
        # if xi_pred.shape[-1] == 2:
        #     mid = np.mean(xi_pred[:, 1])
        #     dev = np.std(xi_pred[:, 1])
        #     cutoff = mid + 2 * dev
        #     is_outside = xi_pred[:, 1] > cutoff
        #     ax.axhline(mid, color="blue", ls="-")
        #     ax.axhline(cutoff, color="blue", ls=":")
        #     if np.sum(is_outside) >= 3:
        #         lib.plotting.mark_trues(
        #             is_outside, color="red", alpha=0.1, ax=ax
        #         )

    plt.tight_layout()
    svg_write(fig)


@timeit
@st.cache(suppress_st_warning=True)
def _dendrogram_trace_plot(X, cluster_labels, cluster_centers):
    """
    Plots dendrogram (left) and mean traces (right)

    Args:
        X (np.ndarray)
        cluster_labels (np.ndarray)
        cluster_centers (np.ndarray)
    """
    n_timeseries = len(set(cluster_labels))

    fig, axes = lib.plotting.dendrogram_ts_layout(n_timeseries=n_timeseries)

    z = lib.math.hierachical_linkage(cluster_centers)
    d = hierarchy.dendrogram(
        z,
        ax=axes[0],
        orientation="left",
        color_threshold=0,
        above_threshold_color="black",
    )
    mean_trace_idx = np.array(list(reversed(d["ivl"])), dtype=int)

    traces_groups = _resample_clustered_traces(
        X=X, cluster_labels=cluster_labels, resample_length=150,
    )

    for idx in mean_trace_idx:
        ax = axes[idx + 1]  # skip first (dendrogram)
        traces = traces_groups[idx]
        clrs = ["black", "red", "blue"]
        channels = traces.shape[-1]
        for c in range(channels):
            t = traces[..., c]
            lib.plotting.plot_timeseries_percentile(
                t,
                ax=ax,
                color=clrs[c],
                min_percentile=50 - 15,
                max_percentile=50 + 15,
                n_percentiles=10,
            )
    svg_write(fig)


@st.cache
def _resample_clustered_traces(X, cluster_labels, resample_length):
    """
    Plots mean and std of resampled traces for each cluster.

    Args:
        X (np.ndarray)
        cluster_labels (np.ndarray)
    """
    traces = []
    for i in range(len(set(cluster_labels))):
        (label_idx,) = np.where(cluster_labels == i)
        t = _resample_traces(
            X[label_idx], length=resample_length, normalize=True,
        )
        traces.append(t)
    return traces


def _plot_length_dist(cluster_lengths, colors, single=False):
    """
    Plots length distributions of each cluster.

    Args:
        cluster_lengths (list of lists of int)
        colors (list of str)
    """
    # max_len = np.array(cluster_lengths).ravel().max()
    bins = np.arange(0, 200, 10)

    if single:
        fig, ax = plt.subplots()
        sns.distplot(
            cluster_lengths, ax=ax, bins=bins, color=colors[0], norm_hist=False
        )
        ax.set_xlabel("Length")
        ax.set_ylabel("Number of samples")
        svg_write(fig)
    else:
        for ij in lib.utils.pairwise_range(cluster_lengths):
            fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
            for ax, i in zip(axes, ij):
                if i is not None:
                    ax.set_title("{}".format(i))
                    sns.distplot(
                        cluster_lengths[i],
                        bins=bins,
                        kde=False,
                        color=colors[i],
                        ax=ax,
                        norm_hist=False,
                    )
                    plt.semilogy()
                    ax.set_xlabel("Length")
                    ax.set_ylabel("Number of samples")
                    ax.set_xlim(0, 200)
                    ax.set_yticks(())
                else:
                    fig.delaxes(ax)
            svg_write(fig)


def _plot_cluster_label_distribution(cluster_labels):
    """
    Plots histogram of cluster label distribution

    Args:
        cluster_labels (np.array)
    """
    cluster_labels = np.ravel(cluster_labels)
    cluster_labels = cluster_labels[cluster_labels != -1]

    bins = range(min(cluster_labels), max(cluster_labels) + 2)
    fig, ax = plt.subplots(figsize=(7, len(set(cluster_labels)) // 2))
    ax.hist(
        cluster_labels,
        bins=bins,
        color="lightgrey",
        rwidth=0.9,
        orientation="horizontal",
    )
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Cluster ID")

    ax.set_yticks(bins)
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_formatter(FixedFormatter(bins))
    ax.yaxis.set_major_formatter(NullFormatter())

    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_markersize(0)

    svg_write(fig)


def _plot_max_intensity(X, cluster_labels, n_rows_cols, colors, mu, sg):
    """
    Plots mean and std of resampled traces for each cluster.

    Args:
        X (np.ndarray)
        cluster_labels (np.ndarray)
        n_rows_cols (int)
        colors (list of str)
        mu (float or np.ndarray)
        sg (float or np.ndarray)
    """
    fig, axes = plt.subplots(nrows=n_rows_cols, ncols=n_rows_cols)
    axes = axes.ravel()

    Xn_all = [np.max(x * sg + mu) for x in X]
    bins = np.arange(0, np.max(Xn_all), 500)

    for i, ax in enumerate(axes):
        try:
            (label_idx,) = np.where(cluster_labels == i)
            X_stat = np.array(
                [np.max(x * sg + mu, axis=0) for x in X[label_idx]]
            )
            # for c in range(X_stat.shape[-1]):
            sns.distplot(
                X_stat[:, 1], kde=True, ax=ax, color=colors[i], bins=bins
            )
            ax.set_yticks(())
            ax.set_xlabel("Intensity")
            ax.set_title("{}".format(i))
        except IndexError:
            fig.delaxes(ax)
    plt.tight_layout()
    svg_write(fig)


def main():
    """
    Main function. Use return to stop script execution.
    """
    # empty line to make it apparent where the execution
    # starts when debugging
    print()
    np.random.seed(42)

    model_dir = ["None"]
    model_dir += sorted(
        glob(os.path.join(lib.globals.models_dir, "*")), reverse=True
    )
    if model_dir is "None":
        return

    data_dir = sorted(
        glob(os.path.join(lib.globals.data_preprocessed_dir, "*.npz"))
    )

    st_model_path = st.selectbox(
        options=model_dir,
        index=0,
        label="Select model",
        format_func=os.path.basename,
    )

    # Find the dataset that the model was trained on
    default_data_path, default_data_idx = _find_default_data_st_selection(
        model_path=st_model_path, data_dir=data_dir
    )
    if default_data_path is None:
        return

    model_name = os.path.basename(st_model_path)
    r = re.search(pattern="single=*", string=model_name)
    if r is not None:
        single_feature = int(model_name[r.span()[1]])
    else:
        single_feature = None

    st_selected_data_path = st.selectbox(
        label="Select dataset to predict on\n(default: {})".format(
            default_data_path
        ),
        options=data_dir,
        index=default_data_idx,
        format_func=os.path.basename,
    )

    st.write("**Model**:")
    st.code(model_name)
    st.write("**Dataset**:")
    st.code(st_selected_data_path)

    if not st.checkbox(label="Check after setting paths", key="st_checkbox_1"):
        return

    st_selected_data_name = os.path.basename(st_selected_data_path)

    X = _load_npz(st_selected_data_path)
    _, _, mu_train, sg_train = _load_train_test_info(path=st_model_path)

    X_true, idx_true, mu_train, sg_train = _prepare_data(
        X=X, mu_train=mu_train, sigma_train=sg_train
    )

    encoding_savename = model_name + "___pred__" + st_selected_data_name

    X_true, X_pred, encodings, mse, indices = _predict(
        X_true=X_true,
        idx_true=idx_true,
        model_path=st_model_path,
        savename=encoding_savename,
        single_feature=single_feature,
    )

    try:
        df = _load_df(st_selected_data_path[:-4] + ".h5", index=indices)
        if "source" in df.columns:
            multi_index_names = list(df["source"].unique())
            if len(multi_index_names) > 1:
                st.write(
                    "**Multi-index file found**\nDataset is composed of the following:"
                )
                for name in multi_index_names:
                    st.code(name)
    except FileNotFoundError:
        pass

    cluster_savename = (
        model_name + "___clust__" + st_selected_data_name + "__cidx.h5"
    )

    umap_savename = model_name + "___umap__" + st_selected_data_name

    # Calculate UMAP embedding
    umap_enc = _umap_embedding(
        features=encodings, embed_into_n_components=2, savename=umap_savename
    )
    umap_enc_orig = umap_enc.copy()

    # Original number of samples, before filtering
    len_X_true_prefilter = len(X_true)

    # Keep only datapoints with len above a minimum length
    arr_lens = np.array([len(xi) for xi in X_true])

    # UMAP generated from whole manifold, but cluster only above certain length
    st_min_length = st.sidebar.number_input(
        min_value=1,
        max_value=100,
        value=20,
        label="Minimum length of data to cluster",
        key="st_min_length",
    )

    st_max_mse = st.sidebar.number_input(
        min_value=0.0,
        max_value=max(mse),
        value=max(mse),
        label="Maximum error of data to cluster",
        key="st_max_mse",
    )

    pre_filtered = np.zeros(len(umap_enc_orig)).astype(int)
    pre_filtered.fill(0)

    # Remove all traces below minimum length from clustering
    (len_above_idx,) = np.where(arr_lens >= st_min_length)
    X_true, X_pred, encodings, umap_enc, mse, indices, = get_index(
        (X_true, X_pred, encodings, umap_enc, mse, indices), index=len_above_idx
    )
    pre_filtered[len_above_idx] = 1

    # Remove all traces above max error from clustering
    (mse_below_idx,) = np.where(mse <= st_max_mse)
    X_true, X_pred, encodings, umap_enc, mse, indices, = get_index(
        (X_true, X_pred, encodings, umap_enc, mse, indices), index=mse_below_idx
    )
    pre_filtered[mse_below_idx] = 1

    pca, explained_var = _pca(features=encodings, embed_into_n_components=None)
    _plot_explained_variance(explained_variance=explained_var)

    clustering_methods = ("Gaussian Mixture Model", "K-means", "HDBSCAN + UMAP")

    st_clust_method = st.sidebar.radio(
        label="Clustering method", options=clustering_methods, index=0
    )

    if st_clust_method == clustering_methods[0]:
        st_clust_n = st.sidebar.number_input(
            label="Number of clusters", value=3
        )
        clabels, centers = _cluster_kmeans(
            features=umap_enc, n_clusters=st_clust_n
        )

    elif st_clust_method == clustering_methods[1]:
        st_clust_n = st.sidebar.number_input(
            label="Number of clusters", value=3
        )
        clabels, centers = _cluster_gmm(
            features=umap_enc, n_clusters=st_clust_n
        )

    elif st_clust_method == clustering_methods[2]:
        st_clust_minsize = st.sidebar.number_input(
            label="Minimum cluster size", value=300
        )
        st_clust_mergethres = st.sidebar.number_input(
            label="Merge threshold (Increase to merge)",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
        )
        clabels, centers = _cluster_hdbscan(
            features=umap_enc,
            min_cluster_size=st_clust_minsize,
            merge_limit=st_clust_mergethres,
        )

    else:
        raise NotImplementedError

    _plot_cluster_label_distribution(cluster_labels=clabels)
    _plot_length_dist(
        cluster_lengths=arr_lens,
        colors=lib.plotting.get_colors("viridis", n_colors=1),
        single=True,
    )
    st.subheader("UMAP displaying length cutoff filter")
    _plot_scatter(
        features=umap_enc_orig, cluster_labels=pre_filtered, density=True
    )

    st.subheader("UMAP displaying density map")
    _plot_scatter(features=umap_enc, density=True, scalebar=True)

    st.subheader("UMAP after clustering on filtered")
    _plot_scatter(features=umap_enc, cluster_labels=clabels, density=True)

    st.subheader("PCA after clustering on filtered")
    _plot_scatter(
        features=pca[:, [0, 1]], cluster_labels=clabels,
    )

    st.subheader("Euclidian distance relationship")
    umap_centers = lib.math.cluster_centers(
        features=umap_enc[clabels != -1], labels=clabels[clabels != -1]
    )
    _dendrogram_trace_plot(
        X=X_true[clabels != -1],
        cluster_labels=clabels[clabels != -1],
        cluster_centers=umap_centers,
    )

    n_clusters = len(set(clabels[clabels != -1]))
    len_X_true_postfilter = len(X_true[clabels != -1])
    colormap = lib.plotting.get_colors("viridis", n_colors=n_clusters)

    st.subheader(
        "Total number of traces in dataset\n(Tracks also removed during clustering)"
    )
    st.write(
        "**Pre-filter:** N = {}\n"
        "**Post-filter:** N = {}\n"
        "**Removed fraction:** {:.2f}".format(
            len_X_true_prefilter,
            len_X_true_postfilter,
            1 - len_X_true_postfilter / len_X_true_prefilter,
        )
    )

    st_nrows = st.sidebar.number_input(
        min_value=2,
        max_value=6,
        value=5,
        label="Rows of traces to show",
        key="st_n_rows",
    )

    st_ncols = st.sidebar.number_input(
        min_value=2,
        max_value=6,
        value=5,
        label="Columns of traces to show",
        key="st_n_cols",
    )

    st_plot_real_values = st.sidebar.checkbox(
        label="Display real values", value=True, key="st_plot_real_values"
    )

    st_separate_y = st.sidebar.checkbox(
        label="Separate y-axis for traces", value=False, key="st_separate_y"
    )

    if not st.checkbox(
        label="Check when you're happy with clustering",
        value=False,
        key="st_checkbox_3",
    ):
        return

    # Iterate over every cluster, plot samples and pick up relevant statistics
    lengths_in_cluster, percentage_of_samples = [], []
    cluster_indexer = []
    for idx in np.unique(clabels):
        if idx == -1:
            continue

        (cidx,) = np.where(clabels == idx)
        percentage = (len(cidx) / len_X_true_postfilter) * 100

        # Index specifics for each cluster label
        len_i = [len(xi) for xi in X_true[cidx]]

        mse_i, X_true_i, X_pred_i, indices_i = get_index(
            (mse, X_true, X_pred, indices), index=cidx
        )

        cluster_indexer.append(
            pd.DataFrame({"cluster": [idx] * len(indices_i), "id": indices_i})
        )

        percentage_of_samples.append(percentage)
        lengths_in_cluster.append(len_i)

        st.subheader(
            "Predictions for {} (N = {} ({:.1f} %))".format(
                idx, len(X_true_i), percentage
            )
        )
        # if X_pred_i[0].shape[-1] == 2:
        #     _clath_aux_peakfinder(X_true_i)

        if len(X_true) < st_nrows * st_ncols:
            nrows = len(X_true)
            ncols = 1
        else:
            nrows = st_nrows
            ncols = st_ncols

        _plot_traces_preview(
            X_true=X_true_i,
            X_pred=X_pred_i,
            sample_indices=indices_i,
            mse=mse_i,
            nrows=nrows,
            ncols=ncols,
            plot_real_values=st_plot_real_values,
            separate_y_ax=st_separate_y,
            mu=mu_train,
            sg=sg_train,
            colors=["black", "red", "blue"],
        )

    cluster_indexer = pd.concat(cluster_indexer, sort=False)

    _save_cluster_sample_indices(
        cluster_sample_indices=cluster_indexer, savename=cluster_savename
    )

    st.subheader("Cluster length distributions")
    _plot_length_dist(
        cluster_lengths=lengths_in_cluster, colors=colormap,
    )


if __name__ == "__main__":
    main()
