import os
import re
from glob import glob
from typing import Iterable

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
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
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.models import Model
from tensorflow.python import keras
from tqdm import tqdm
from umap import UMAP
import pandas as pd

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


def _load_train_test_info(path):
    """
    Load indices for train/test split on the given dataset from a model dir.

    Args:
        path (str)
    """
    f = np.load(os.path.join(path, "info.npz"), allow_pickle=True)
    idx_train, idx_test, mu, sg = f["info"]
    return idx_train, idx_test, mu, sg


def _load_multi_index(data_path, fallback_indices):
    """
    If dataset is composed of several individual datasets
    data1.h5, data2.h5, data3.h5...
    that are to be compared, load the corresponding multi-index so that they
    can be distinguished, and re-found after clustering

    The index is a dataframe with 'file' and 'idx' obtained from the
    ordering of df.groupby(["file", "particle"])

    If no indices are given, the multi index will be constructed from the
    fallback indices, so that indices are always available

    Args:
        data_path (str)
        fallback_indices (np.array)
    """
    try:
        multi_index = pd.DataFrame(pd.read_hdf(data_path[:-4] + "_idx.h5"))
    except FileNotFoundError:
        print("No multi-index file found")
        multi_index = pd.DataFrame({"file": data_path, "idx": fallback_indices})
    multi_index["cluster"] = 0
    return multi_index


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
        idx_train (Union[np.ndarray, None])
        idx_test (Union[np.ndarray, None])
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


def _predict(X_true, idx_true, model_path, savename, single_feature=None):
    """
    Predicts autoencoder features and saves them. Saving as npz is required for
    interactivity, as predictions use the GPU (which may not be available while
    training). Warning: order is not preserved due to batching!

    Args:
        X_true (np.ndarray)
        idx_true (np.ndarray)
        model_path (str)
        savename (str)
        single_feature (Union[None, int])
    """
    # See if there's a cached version
    basepath = "results/extracted_features/{}"

    try:
        f = np.load(basepath.format(savename), allow_pickle=True)
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

        if len(X_true.shape) == 3:
            # If trained with equal lengths normal tensor
            # This also means that order is preserved throughout
            X_true = X_true
            X_pred = autoencoder.predict(X_true)
            features = encoder.predict(X_true)
            mse = lib.math.mean_squared_error(X_true, X_pred, axis=(1, 2))
            indices = idx_true

        else:
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
                max_batch_size=128,
                shuffle_samples=True,
                shuffle_batches=True,
            )

            indices = X_.indices

            X_ = tf.data.Dataset.from_generator(
                generator=X_,
                output_types=(tf.float64, tf.float64),
                output_shapes=(
                    (None, None, n_features),
                    (None, None, n_features),
                ),
            )

            # Predict on batches and unravel for single-item use
            X_true, X_pred, features, mse = [], [], [], []
            for xi_true, _ in tqdm(X_):
                if single_feature is not None:
                    # if single feature, expand last dim to (..., 1)
                    p_xi_true = tf.expand_dims(
                        xi_true[..., single_feature], axis=2
                    )
                else:
                    p_xi_true = xi_true

                xi_pred = autoencoder.predict_on_batch(p_xi_true)
                fi = encoder.predict_on_batch(p_xi_true)
                ei = lib.math.mean_squared_error(
                    p_xi_true.numpy(), xi_pred.numpy(), axis=(1, 2)
                )

                # Make sure they're numpy arrays now and not EagerTensors!
                X_true.extend(
                    np.array(xi_true)
                )  # Append the original one and not the one used for predictions
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
            basepath.format(savename),
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
    umap = UMAP(
        n_neighbors=30, min_dist=0.0, n_components=embed_into_n_components
    )
    manifold = umap.fit_transform(features)
    return manifold


@st.cache
def _cluster(features, savename):
    """
    Performs clustering on given features.

    Args:
        features (np.ndarray)
        savename (Union[str, None])
        pre_trained_path (str)
    """
    basepath = "results/cluster_classifiers/{}"

    clf = hdbscan.HDBSCAN(
        prediction_data=True,
        allow_single_cluster=False,
        approx_min_span_tree=True,
        min_samples=50,
    )
    clf.fit(features)

    # Save fitted clusters for later, but only if they're newly fitted
    if savename is not None:
        np.savez(basepath.format(savename), clf=clf)

    labels, _ = hdbscan.approximate_predict(
        clusterer=clf, points_to_predict=features
    )
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
    # return [a[rand_idx] for a in arrs]


def _save_cluster_sample_indices(cluster_sample_indices, model_name, data_name):
    """
    Save sample indices for clusters to later retrieve from dataset

    Args:
        cluster_sample_indices (pd.DataFrame)
        model_name (str)
        data_name (str)
    """
    idx_savename = model_name + "_" + data_name + "__cidx.h5"
    cluster_sample_indices.to_hdf(
        "results/cluster_indices/{}".format(idx_savename), key="df"
    )


def _find_peaks(X, n_frames=3, n_std=2):
    """
    Finds peaks for input list of arrays.

    Args:
        X (np.ndarray)
        n_frames (int)
        n_std (Union[int, float])
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
        cluster_labels (Union[np.ndarray, None])
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
    sample_files,
    plot_real_values,
    separate_y_ax,
    nrows,
    ncols,
    mu,
    sg,
    colors,
    single_feature=None,
):
    """
    Plots a subset of clustered traces for inspection

    Args:
        X_true (np.ndarray)
        X_pred (np.ndarray)
        mse (np.ndarray)
        sample_indices (np.ndarray)
        sample_files (np.array):
        plot_real_values (bool)
        separate_y_ax (bool)
        nrows (int)
        ncols (int)
        mu (Union[float, np.ndarray])
        sg (Union[float, np.ndarray])
        colors (List[str])
        single_feature (Union[None, bool])

    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        try:
            xi_true = X_true[i]
            xi_pred = X_pred[i]
            ei = mse[i]
            idx = sample_indices[i]
            file = sample_files[i]
        except IndexError:
            fig.delaxes(ax)
            continue

        rng = range(len(xi_true))

        ax.set_title(
            "E = {:.2f}, L = {}\n"
            "idx = {}\n"
            "{}".format(ei, len(xi_true), idx, file),
            fontsize=5,
        )
        ax.set_xticks(())

        if plot_real_values:
            xi_true = xi_true * sg + mu
            if single_feature is not None:
                xi_pred = xi_pred * sg[single_feature] + mu[single_feature]
            else:
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
    X, cluster_labels, cluster_lengths, percentages, n_rows_cols,
):
    """
    Plots mean and std of resampled traces for each cluster.

    Args:
        X (np.ndarray):
        cluster_labels (np.ndarray):
        cluster_lengths (List[int]):
        percentages (List[float]):
        n_rows_cols (int):
    """
    fig, axes = plt.subplots(nrows=n_rows_cols, ncols=n_rows_cols)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        try:
            (label_idx,) = np.where(cluster_labels == i)
            resample_len = np.max(cluster_lengths[i])
            traces = _resample_traces(
                X[label_idx], length=resample_len, normalize=True,
            )
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

            ax.set_xlim(0, resample_len)
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title("{} ({:.1f} %)".format(i, percentages[i]))
        except IndexError:
            fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig)
    # svg_write(fig)


def _plot_length_dist(cluster_lengths, n_rows_cols, colors):
    """
    Plots length distributions of each cluster.

    Args:
        cluster_lengths (List[int]):
        n_rows_cols (int):
        colors (List[str]):


    """
    fig, axes = plt.subplots(nrows=n_rows_cols, ncols=n_rows_cols)
    axes = axes.ravel()
    bins = np.arange(0, 200, 10)
    for i, ax in enumerate(axes):
        try:
            ax.set_title("{}".format(i))
            sns.distplot(
                cluster_lengths[i], bins=bins, kde=True, color=colors[i], ax=ax,
            )
            ax.set_xlabel("Length")
            # ax.set_ylabel("Count")
            ax.set_xlim(20, 200)
            ax.set_yticks(())
        except IndexError:
            fig.delaxes(ax)
    plt.tight_layout()
    svg_write(fig)


def _plot_max_intensity(X, cluster_labels, n_rows_cols, colors, mu, sg):
    """
    Plots mean and std of resampled traces for each cluster.

    Args:
        X (np.ndarray):
        cluster_labels (np.ndarray):
        n_rows_cols (int):
        colors (List[str]):
        mu (Union[float, np.ndarray]):
        sg (Union[float, np.ndarray]):


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
    random_state = st.sidebar.number_input(label="Random seed", value=0)
    np.random.seed(random_state)

    model_dir = sorted(glob("models/*"))
    data_dir = sorted(glob("results/intensities/*.npz"))

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

    # Predict only on the test set (not really using the training set for
    # anything downstream at the moment
    single_feature = re.search("single=.*", st_model_path)
    if single_feature is not None:
        print("Using only single channel")
        single_feature = int(single_feature[0][-1])

    # Always load mu and sg obtained from train set
    idx_train, idx_test, mu_train, sg_train = _load_train_test_info(
        st_model_path
    )

    # Check if selected dataset is the same as used for train/test
    # to provide this option
    st_selected_data_name = os.path.basename(st_selected_data_path)
    if st_selected_data_name == os.path.basename(default_data_path):
        use_data = st.sidebar.radio(
            label="Dataset to use for predictions",
            options=["all", "test", "train"],
            index=0,
        )

        # fitted traces for autoencoder predictions
        X = _load_npz(st_selected_data_path)  # type: np.ndarray

        X_train, X_test = _standardize_train_test(
            X=X,
            idx_train=idx_train,
            idx_test=idx_test,
            mu=mu_train,
            sigma=sg_train,
        )

        if use_data == "test":
            X_true = X_test
            idx_true = idx_test
        elif use_data == "train":
            X_true = X_train
            idx_true = idx_test
        elif use_data == "all":
            X_true = np.concatenate((X_train, X_test))
            idx_true = np.concatenate((idx_train, idx_test))
        else:
            raise ValueError("Invalid data selection")
    else:
        st.subheader("**External dataset chosen**")
        # Take a different dataset without train/test split available
        X_true = _load_npz(st_selected_data_path)
        X_true = _standardize_train_test(X_true, mu=mu_train, sigma=sg_train)
        idx_true = np.array(range(len(X_true)))

    X_true, X_pred, features, mse, index_df = _predict(
        X_true=X_true,
        idx_true=idx_true,
        model_path=st_model_path,
        single_feature=single_feature,
        savename=model_name + "___pred__" + st_selected_data_name,
    )
    mse_orig = mse.copy()
    index_df = _load_multi_index(
        data_path=st_selected_data_path, fallback_indices=index_df
    )

    multi_index_names = list(index_df["file"].unique())
    if len(set(multi_index_names)) > 1:
        st.write(
            "**Multi-index file found**\nDataset is composed of the following:"
        )
        for name in multi_index_names:
            st.code(name)

    st_subsample_frac = st.sidebar.number_input(
        min_value=0.01,
        max_value=1.0,
        value=0.01,
        step=0.01,
        label="Percentage of original dataset to use",
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

    X_true, X_pred, features, mse, index_df = _random_subset(
        (X_true, X_pred, features, mse, index_df,),
        n_samples=int(len(X_true) * st_subsample_frac),
    )
    len_X_true_prefilter = len(X_true)

    # Keep only datapoints with len above a minimum length
    arr_lens = np.array([len(xi) for xi in X_true])
    (len_above_idx,) = np.where(arr_lens >= st_min_length)
    X_true, X_pred, features, mse, index_df = get_index(
        (X_true, X_pred, features, mse, index_df), index=len_above_idx
    )

    # Keep only datapoints with error below threshold and cluster these
    # (high error data shouldn't be there)
    (errorbelow_idx,) = np.where(mse < st_mse_filter)
    X_true, X_pred, features, mse, index_df = get_index(
        (X_true, X_pred, features, mse, index_df), index=errorbelow_idx
    )

    pca_raw, _ = _pca(features, embed_into_n_components=2)
    st.subheader("PCA of raw datapoints")
    _plot_scatter(features=pca_raw)

    clust_savename = model_name + "__" + st_selected_data_name

    st_cluster_on = st.sidebar.radio(
        options=["raw", "umap"], index=0, label="Type of feature to cluster"
    )
    if st_cluster_on == "umap":
        c_features = _umap_embedding(
            features=features, embed_into_n_components=2
        )
    elif st_cluster_on == "raw":
        c_features = features
    else:
        raise NotImplementedError

    cluster_labels_w_outliers = _cluster(
        features=c_features, savename=clust_savename,
    )

    # perform both PCA and manifold of features
    st.subheader("UMAP embedding")
    _plot_scatter(features=c_features, cluster_labels=cluster_labels_w_outliers)

    # Remove HDBSCAN outliers before any further analysis
    (cluster_labels_idx,) = np.where(cluster_labels_w_outliers != -1)

    (X_true, X_pred, features, mse, cluster_labels, index_df,) = get_index(
        (X_true, X_pred, features, mse, cluster_labels_w_outliers, index_df,),
        index=cluster_labels_idx,
    )

    index_df["cluster"] = cluster_labels
    n_clusters = len(set(cluster_labels))
    len_X_true_postfilter = len(X_true)
    n_rows_cols = int(np.ceil(np.sqrt(n_clusters)))
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
    # for later
    # cluster_sample_indices = []
    lengths_in_cluster, percentage_of_samples = [], []
    for i in range(len(set(cluster_labels))):
        (idx,) = np.where(cluster_labels == i)
        percentage = (len(idx) / len_X_true_postfilter) * 100

        if percentage < 1:
            "_Cluster {} contains less than 1% of samples_".format(i)

        len_i = [len(xi) for xi in X_true[idx]]
        mse_i = mse[idx]
        X_true_i = X_true[idx]
        X_pred_i = X_pred[idx]

        sample_indices_i = index_df["idx"].values[idx]
        sample_files_i = index_df["file"].values[idx]

        percentage_of_samples.append(percentage)
        lengths_in_cluster.append(len_i)

        # index_df["cluster"].iloc[idx] = i
        # cluster_sample_indices.append(index_df.iloc[idx])

        st.subheader(
            "Predictions for {} (N = {} ({:.1f} %))".format(
                i, len(X_true_i), percentage
            )
        )

        if X_pred_i[0].shape[-1] == 2:
            # Clathrin/auxilin specific block
            has_peak, has_no_peak = _find_peaks(X_pred_i)
            st.write("{} traces with peaks".format(has_peak))
            st.write("{} traces with no peaks".format(has_no_peak))
            st.write(
                "{:.1f} % of traces have peaks".format(
                    has_peak / (has_peak + has_no_peak) * 100
                )
            )

        if len(X_true) < st_nrows * st_ncols:
            nrows = len(X_true)
            ncols = 1
        else:
            nrows = st_nrows
            ncols = st_ncols

        _plot_traces_preview(
            X_true=X_true_i,
            X_pred=X_pred_i,
            sample_indices=sample_indices_i,
            sample_files=sample_files_i,
            mse=mse_i,
            nrows=nrows,
            ncols=ncols,
            plot_real_values=st_plot_real_values,
            separate_y_ax=st_separate_y,
            mu=mu_train,
            sg=sg_train,
            single_feature=single_feature,
            colors=["black", "red"],
        )

    # cluster_sample_indices = pd.concat(cluster_sample_indices)
    st.write(index_df.sort_values(by="idx").head())

    _save_cluster_sample_indices(
        cluster_sample_indices=index_df,
        model_name=model_name,
        data_name=st_selected_data_name,
    )
    quit()

    st.subheader("Cluster length distributions")
    _plot_length_dist(
        cluster_lengths=lengths_in_cluster,
        n_rows_cols=n_rows_cols,
        colors=colormap,
    )

    st.subheader("Mean (resampled) **true** traces in each cluster")
    _plot_mean_trace(
        X=X_true,
        cluster_labels=cluster_labels,
        cluster_lengths=lengths_in_cluster,
        n_rows_cols=n_rows_cols,
        percentages=percentage_of_samples,
    )

    # st.subheader("Mean (resampled) **precited** traces in each cluster")
    # _plot_mean_trace(
    #     X=X_pred,
    #     cluster_labels=cluster_labels,
    #     cluster_lengths=lengths_in_cluster,
    #     n_rows_cols=n_rows_cols,
    #     percentages=percentage_of_samples,
    # )


if __name__ == "__main__":
    main()
