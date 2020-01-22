import os
import re
from glob import glob
from typing import Iterable

import hdbscan
import matplotlib.pyplot as plt
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
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.models import Model
from tensorflow.python import keras
from tqdm import tqdm

import lib.globals
import lib.math
import lib.models
import lib.plotting
import lib.utils
from lib.plotting import svg_write
from lib.tfcustom import VariableTimeseriesBatchGenerator, gelu
from lib.utils import get_index

sns.set_style("dark")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}


@st.cache
def _load_npz(path):
    """
    Loads all traces

    Args:
        path (str)
    """
    return np.load(path, allow_pickle=True)["data"]


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


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
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


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def _standardize_single(X, mu, sigma):
    """
    Normalizes a single array

    Args:
        X (np.array)
        mu (float)
        sigma (float)

    """
    return lib.math.standardize(X, mu, sigma)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def _prepare_data(X, model_dir, data_name, data_path, set_type):
    """
    Args:
        X (np.array)
        data_name (str)
        data_path (str)
        set_type (str)
    """
    # Always load mu and sg obtained from train set
    idx_train, idx_test, mu_train, sg_train = _load_train_test_info(model_dir)

    if data_name != os.path.basename(data_path):
        st.subheader("**External dataset chosen**")
        X_true = _standardize_single(X=X, mu=mu_train, sigma=sg_train)
        idx_true = np.array(range(len(X_true)))
    else:
        # fitted traces for autoencoder predictions
        X_train, X_test = _standardize_train_test(
            X=X,
            idx_train=idx_train,
            idx_test=idx_test,
            mu=mu_train,
            sigma=sg_train,
        )

        if set_type in ("train", "test"):
            if set_type == "train":
                X_true = X_train
                idx_true = idx_test
            else:
                X_true = X_test
                idx_true = idx_test
        else:
            X_true = np.concatenate((X_train, X_test))
            idx_true = np.concatenate((idx_train, idx_test))

    return X_true, idx_true, mu_train, sg_train


def _latest_model(model_dir, recency=1):
    """
    Fetches latest model in directory.

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
    autoencoder, encoding_layer_names=("encoded", "z_sample"),
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


def _predict(X_true, idx_true, model_path, savename, zero_feature):
    """
    Predicts autoencoder features and saves them. Saving as npz is required for
    interactivity, as predictions use the GPU (which may not be available while
    training). Warning: order is not preserved due to batching!

    Args:
        X_true (np.ndarray)
        idx_true (np.ndarray)
        model_path (str)
        savename (str)
        zero_feature (int or None)
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
            latest_model_path, custom_objects={"gelu": gelu}
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
            if zero_feature is not None:
                _xi_true = xi_true
                _xi_true = _xi_true.numpy()
                _xi_true[..., 1] = 0
            else:
                _xi_true = xi_true
            fi = encoder.predict_on_batch(_xi_true)

            # Predict reconstruction
            xi_pred = autoencoder.predict_on_batch(_xi_true)

            # Calculate error of reconstruction
            ei = lib.math.mean_squared_error(
                xi_true.numpy(), xi_pred.numpy(), axis=(1, 2)
            )

            # Make sure they're numpy arrays now and not EagerTensors!
            X_true.extend(np.array(xi_true))
            X_pred.extend(np.array(xi_pred))
            features.extend(np.array(fi))
            mse.extend(np.array(ei))

        X_true, X_pred, features, mse = map(
            np.array, (X_true, X_pred, features, mse)
        )

    # Shuffle data to be certain that all plots show random samples
    X_true, X_pred, features, mse, indices = sklearn.utils.shuffle(
        X_true, X_pred, features, mse, indices
    )
    np.savez(
        os.path.join(lib.globals.encodings_dir, savename),
        X_true=X_true,
        X_pred=X_pred,
        features=features,
        mse=mse,
        indices=indices,
    )
    print("File saved as {}".format(savename))

    if not st._is_running_with_streamlit:
        print(
            "Predictions done. You can stop the script now and run streamlit to visualize."
        )

    return X_true, X_pred, features, mse, indices


@st.cache
def _pca(features, embed_into_n_components):
    """
    Calculates the PCA of raw input features.

    Args:
        features (np.ndarray)
        embed_into_n_components (int)
    """
    pca = sklearn.decomposition.PCA(n_components=embed_into_n_components)
    pca_features = pca.fit_transform(features)
    explained_var = np.cumsum(np.round(pca.explained_variance_ratio_, 3))
    return pca_features, explained_var


@st.cache
def _umap_embedding(features, embed_into_n_components=2):
    """
    Calculates the UMAP embedding of raw input features for visualization.

    Args:
        features (np.ndarray)
        embed_into_n_components (int)
    """

    # features = sklearn.decomposition.PCA(n_components = 10).fit_transform(features)
    spread = np.std(features, axis=(0, 1))

    u = umap.UMAP(
        n_components=embed_into_n_components,
        random_state=42,
        spread=spread,
        n_neighbors=10,
        min_dist=0.2,
        init="random",
        learning_rate=0.5,
        n_epochs=1000,
    )
    return u.fit_transform(features)


@st.cache
def _cluster(features, min_cluster_size=30, min_core_points=0):
    """
    Cluster points using HDBSCAN.

    Args:
        features (np.array)
        min_cluster_size (int)
        min_core_points (int)
    """
    clf = hdbscan.HDBSCAN(
        prediction_data=True,
        allow_single_cluster=False,
        approx_min_span_tree=True,
        min_samples=min_core_points,
        min_cluster_size=min_cluster_size,
        core_dist_n_jobs=-1,
    )
    labels = clf.fit_predict(features)
    return labels


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
            t = trace / trace.max(axis=0)
            t -= np.min(t, axis=0)
            new.append(t)

        traces_re = np.array(new)
    return traces_re


@st.cache
def _random_subset(list_of_objs, n_samples):
    """
    Subsamples arrays

    Args:
        n_samples (int)
    """
    lens = [len(a) for a in list_of_objs]
    if not lib.utils.all_equal(lens):
        raise ValueError("All arrays must contain same number of samples")

    idx = np.arange(0, lens[0], 1)
    rand_idx = np.random.choice(idx, n_samples, replace=False)

    return lib.utils.get_index(list_of_objs, index=rand_idx)


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
        X (np.array):
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


def _plot_scatter(features, cluster_labels=None):
    """
    Plots lower dimensional embedding with predicted cluster labels.

    Args:
        features (np.ndarray)
        cluster_labels (np.ndarray or None)
    """
    fig, ax = plt.subplots()

    if cluster_labels is None:
        ax.scatter(
            features[:, 0], features[:, 1], color="black", s=10, alpha=0.5,
        )
    else:
        lmin = min(cluster_labels[cluster_labels != -1])
        lmax = max(cluster_labels[cluster_labels != -1])

        mids = []
        for i in range(len(set(cluster_labels))):
            emb_i = features[cluster_labels == i]
            mids.append(np.mean(emb_i, axis=0))

        cmap = plt.get_cmap("magma", lmax - lmin + 1)

        ax.scatter(
            features[:, 0][cluster_labels == -1],
            features[:, 1][cluster_labels == -1],
            color="grey",
            s=10,
            alpha=0.1,
        )

        cax = ax.scatter(
            features[:, 0][cluster_labels != -1],
            features[:, 1][cluster_labels != -1],
            c=cluster_labels[cluster_labels != -1],
            cmap=cmap,
            s=10,
            alpha=0.8,
        )
        fig.colorbar(cax, ax=ax, ticks=np.arange(lmin, lmax + 1))

        for i, m in enumerate(mids):
            ax.annotate(
                xy=m[[0, 1]],
                s=i,
                bbox=dict(boxstyle="square", fc="w", ec="grey", alpha=0.9),
            )

    ax.set_xlabel("C0")
    ax.set_ylabel("C1")
    plt.tight_layout()
    st.write(fig)


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
            "E = {:.2f}, L = {}\n"
            "idx = {}\n".format(ei, len(xi_true), idx),
            fontsize=5,
        )
        ax.set_xticks(())

        if plot_real_values:
            xi_true = xi_true * sg + mu
            xi_pred = xi_pred * sg + mu

        for c in range(xi_true.shape[-1]):
            if not separate_y_ax:
                ax.plot(rng, xi_true[:, c], alpha=0.3, color=colors[c])
                try:
                    ax.plot(rng, xi_pred[:, c], color=colors[c])
                except IndexError:
                    pass
            else:
                ax_ = ax.twinx()
                ax_.plot(rng, xi_true[:, c], alpha=0.3, color=colors[c])
                try:
                    ax_.plot(rng, xi_pred[:, c], color=colors[c])
                except IndexError:
                    pass

                ax_.set_yticks(())
                ax.set_yticks(())

        # Works only for 2C clathrin/auxilin traces
        if xi_pred.shape[-1] == 2:
            mid = np.mean(xi_pred[:, 1])
            dev = np.std(xi_pred[:, 1])
            cutoff = mid + 2 * dev
            is_outside = xi_pred[:, 1] > cutoff
            ax.axhline(mid, color="blue", ls="-")
            ax.axhline(cutoff, color="blue", ls=":")
            if np.sum(is_outside) >= 3:
                lib.plotting.mark_trues(
                    is_outside, color="red", alpha=0.1, ax=ax
                )

    plt.tight_layout()
    svg_write(fig)


def _plot_mean_trace(
    X, cluster_labels, cluster_lengths, percentages,
):
    """
    Plots mean and std of resampled traces for each cluster.

    Args:
        X (np.ndarray)
        cluster_labels (np.ndarray)
        cluster_lengths (list of lists of int)
        percentages (list of float)
    """
    for ij in lib.utils.pairwise_range(cluster_lengths):
        fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

        for ax, i in zip(axes, ij):
            if i is not None:
                (label_idx,) = np.where(cluster_labels == i)
                resample_len = np.max(cluster_lengths[i])
                traces = _resample_traces(
                    X[label_idx], length=resample_len, normalize=True,
                )

                ax.set_xlim(0, resample_len)
                ax.set_xticks(())
                ax.set_yticks(())
                ax.set_title("{} ({:.1f} %)".format(i, percentages[i]))

                clrs = ["black", "red"]
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
            else:
                fig.delaxes(ax)
        svg_write(fig)


def _plot_length_dist(cluster_lengths, colors):
    """
    Plots length distributions of each cluster.

    Args:
        cluster_lengths (list of lists of int)
        colors (list of str)
    """
    bins = np.arange(0, 200, 10)

    for ij in lib.utils.pairwise_range(cluster_lengths):
        fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
        for ax, i in zip(axes, ij):
            if i is not None:
                ax.set_title("{}".format(i))
                sns.distplot(
                    cluster_lengths[i],
                    bins=bins,
                    kde=True,
                    color=colors[i],
                    ax=ax,
                )
                ax.set_xlabel("Length")
                ax.set_xlim(0, 200)
                ax.set_yticks(())
            else:
                fig.delaxes(ax)
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
    st_random_state = st.sidebar.number_input(label="Random seed", value=0)
    np.random.seed(st_random_state)

    model_dir = sorted(glob(os.path.join(lib.globals.models_dir, "*")))
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

    # Check if selected dataset is the same as used for train/test
    # to provide this option
    st_data_type = st.sidebar.radio(
        label="Dataset to use for predictions",
        options=["all", "test", "train"],
        index=0,
    )

    st_selected_data_name = os.path.basename(st_selected_data_path)

    X = _load_npz(st_selected_data_path)
    X_true, idx_true, mu_train, sg_train = _prepare_data(
        X=X,
        data_name=st_selected_data_name,
        data_path=default_data_path,
        model_dir=st_model_path,
        set_type=st_data_type,
    )

    st_set_zero = st.sidebar.radio(
        options=["None", "1", "2"],
        label="Select feature to set to 0 (thus ignoring it)",
        index=0,
        key="st_set_zero",
    )
    if st_set_zero != "None":
        encoding_savename = (
            model_name + "___pred__" + st_set_zero + st_selected_data_name
        )
        cluster_savename = (
            model_name
            + "___clust__"
            + st_set_zero
            + st_selected_data_name
            + "__cidx.h5"
        )
        st_set_zero = int(st_set_zero)
    else:
        encoding_savename = model_name + "___pred__" + st_selected_data_name
        cluster_savename = (
            model_name
            + "___clust__"
            + st_set_zero
            + st_selected_data_name
            + "__cidx.h5"
        )
        st_set_zero = None

    X_true, X_pred, encodings, mse, indices = _predict(
        X_true=X_true,
        idx_true=idx_true,
        model_path=st_model_path,
        savename=encoding_savename,
        zero_feature=st_set_zero,
    )
    mse_orig = mse.copy()

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
        df = None

    st_subsample_frac = st.sidebar.number_input(
        min_value=0.01,
        max_value=1.0,
        value=0.01,
        step=0.01,
        label="Fraction of original dataset to use",
        key="st_subsample_frac",
    )

    st_min_length = st.sidebar.number_input(
        min_value=1,
        max_value=100,
        label="Minimum length of data to cluster",
        key="st_min_length",
    )

    st_mse_filter = st.sidebar.number_input(
        max_value=np.max(mse_orig),
        min_value=np.min(mse_orig),
        value=np.max(mse_orig),
        step=np.max(mse_orig) / 100,
        format="%0.2f",
        label="Maximum error of data to cluster",
    )

    if not st.checkbox(
        label="Check after setting sidebar parameters",
        value=False,
        key="st_checkbox_2",
    ):
        return

    # Pick a random subset to speed up computation
    X_true, X_pred, encodings, mse, indices = _random_subset(
        (X_true, X_pred, encodings, mse, indices),
        n_samples=int(len(X_true) * st_subsample_frac),
    )
    len_X_true_prefilter = len(X_true)

    # Keep only datapoints with len above a minimum length
    arr_lens = np.array([len(xi) for xi in X_true])
    (len_above_idx,) = np.where(arr_lens >= st_min_length)

    X_true, X_pred, encodings, mse, indices = get_index(
        (X_true, X_pred, encodings, mse, indices), index=len_above_idx
    )

    # Keep only datapoints with error below threshold and cluster these
    # (high error data shouldn't be there)
    (errorbelow_idx,) = np.where(mse < st_mse_filter)
    X_true, X_pred, encodings, mse, indices = get_index(
        (X_true, X_pred, encodings, mse, indices), index=errorbelow_idx
    )

    st_cluster_on = st.sidebar.radio(
        options=["umap", "raw"], index=0, label="Type of feature to cluster"
    )
    if st_cluster_on == "umap":
        c_encodings = _umap_embedding(
            features=encodings,
            embed_into_n_components=2,
        )
    elif st_cluster_on == "raw":
        c_encodings = encodings
    else:
        raise NotImplementedError

    st_clust_min_size = st.sidebar.number_input(
        label="Min cluster size", value=30
    )
    st_clust_min_samples = st.sidebar.number_input(
        label="Min number of core points", value=10
    )

    cluster_labels_w_outliers = _cluster(
        features=c_encodings,
        min_cluster_size=st_clust_min_size,
        min_core_points=st_clust_min_samples,
    )

    st.subheader("PCA")
    pca_raw, _ = _pca(encodings, embed_into_n_components=2)
    _plot_scatter(features=encodings, cluster_labels=cluster_labels_w_outliers)

    st.subheader("UMAP")
    _plot_scatter(
        features=c_encodings, cluster_labels=cluster_labels_w_outliers
    )

    # Remove HDBSCAN outliers before any further analysis
    (cluster_labels_idx,) = np.where(cluster_labels_w_outliers != -1)

    (X_true, X_pred, encodings, mse, cluster_labels, indices,) = get_index(
        (X_true, X_pred, encodings, mse, cluster_labels_w_outliers, indices,),
        index=cluster_labels_idx,
    )

    n_clusters = len(set(cluster_labels))
    len_X_true_postfilter = len(X_true)
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

    st_nrows = st.sidebar.slider(
        min_value=2,
        max_value=6,
        value=4,
        label="Rows of traces to show",
        key="st_n_rows",
    )

    st_ncols = st.sidebar.slider(
        min_value=2,
        max_value=6,
        value=4,
        label="Columns of traces to show",
        key="st_n_cols",
    )

    st_plot_real_values = st.sidebar.checkbox(
        label="Display real values", value=True, key="st_plot_real_values"
    )

    st_separate_y = st.sidebar.checkbox(
        label="Separate y-axis for traces", value=False, key="st_separate_y"
    )

    # Iterate over every cluster, plot samples and pick up relevant statistics
    lengths_in_cluster, percentage_of_samples = [], []
    cluster_indexer = []
    for i in range(len(set(cluster_labels))):
        (cidx,) = np.where(cluster_labels == i)
        percentage = (len(cidx) / len_X_true_postfilter) * 100

        # Index specifics for each cluster label
        len_i = [len(xi) for xi in X_true[cidx]]

        mse_i, X_true_i, X_pred_i, indices_i = get_index(
            (mse, X_true, X_pred, indices), index=cidx
        )

        # TODO: utilize sub_id
        if df is not None:
            sub_df = df[df["id"].isin(indices_i)]
            sub_file_idx = sub_df["sub_id"].unique().values
            st.write(sub_file_idx)
            if not len(sub_file_idx) == X_true_i:
                raise ValueError
        else:
            sub_file_idx = None

        cluster_indexer.append(
            pd.DataFrame({"cluster": [i] * len(indices_i), "id": indices_i})
        )

        percentage_of_samples.append(percentage)
        lengths_in_cluster.append(len_i)

        st.subheader(
            "Predictions for {} (N = {} ({:.1f} %))".format(
                i, len(X_true_i), percentage
            )
        )
        if X_pred_i[0].shape[-1] == 2:
            _clath_aux_peakfinder(X_true_i)

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
            colors=["black", "red"],
        )

    cluster_indexer = pd.concat(cluster_indexer, sort=False)

    _save_cluster_sample_indices(
        cluster_sample_indices=cluster_indexer, savename=cluster_savename
    )

    st.subheader("Cluster length distributions")
    _plot_length_dist(
        cluster_lengths=lengths_in_cluster, colors=colormap,
    )

    st.subheader("Mean (resampled) **true** traces in each cluster")
    _plot_mean_trace(
        X=X_true,
        cluster_labels=cluster_labels,
        cluster_lengths=lengths_in_cluster,
        percentages=percentage_of_samples,
    )

    st.subheader("Mean (resampled) **precited** traces in each cluster")
    _plot_mean_trace(
        X=X_pred,
        cluster_labels=cluster_labels,
        cluster_lengths=lengths_in_cluster,
        percentages=percentage_of_samples,
    )


if __name__ == "__main__":
    main()
