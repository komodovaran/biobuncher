import os
import re
from collections import namedtuple
from glob import glob
from typing import Any, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.manifold
import sklearn.metrics
import sklearn.mixture
import sklearn.model_selection
import sklearn.preprocessing
import streamlit as st
import tensorflow as tf
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from tensorflow.keras.models import Model
from tensorflow.python import keras
from tqdm import tqdm
import parmap
from umap import UMAP
import time
import lib.math
import lib.models
import lib.plotting
import lib.utils
from lib.plotting import svg_write
from lib.tfcustom import VariableTimeseriesBatchGenerator, gelu
from lib.utils import get_index

sns.set_style("dark")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}


def _setup_streamlit_widgets(
    n_components_max: int, original_mse: np.array
) -> Any:
    """
    Sets up all the streamlit widgets and returns their values.
    """
    StWidgets = namedtuple(
        "StWidgets",
        [
            "min_length",
            "feature_type",
            "n_clusters",
            "n_components",
            "mse_filter",
            "confidence_filter",
            "plot_real_values",
            "separate_y",
        ],
    )

    StWidgets.min_length = st.sidebar.slider(
        min_value=1, max_value=100, label="Minimum length of data to cluster"
    )

    StWidgets.feature_type = st.sidebar.radio(
        options=["raw", "pca"], label="Type of features to cluster on"
    )

    StWidgets.n_clusters = st.sidebar.slider(
        value=2, min_value=1, max_value=32, label="Number of clusters"
    )

    StWidgets.n_components = st.sidebar.slider(
        value=2,
        min_value=2,
        max_value=n_components_max,
        label="Components for PCA embedding",
    )

    StWidgets.mse_filter = st.sidebar.number_input(
        max_value=np.max(original_mse),
        min_value=np.min(original_mse),
        value=np.max(original_mse),
        step=np.max(original_mse) / 100,
        format="%0.2f",
        label="Cluster only traces with error below:",
    )

    StWidgets.confidence_filter = st.sidebar.number_input(
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        format="%0.2f",
        label="After clustering, Keep only traces with confidence above:",
    )
    StWidgets.nrows = st.sidebar.slider(
        min_value=2, max_value=6, value=4, label="Rows of traces to show"
    )

    StWidgets.ncols = st.sidebar.slider(
        min_value=2, max_value=6, value=4, label="Columns of traces to show"
    )

    StWidgets.plot_real_values = st.sidebar.checkbox(
        label="Display real values", value=True
    )

    StWidgets.separate_y = st.sidebar.checkbox(
        label="Separate y-axis for traces", value=False
    )

    return StWidgets


def _get_encoding_layer(
    autoencoder: Model,
    encoding_layer_names: Iterable[str] = ("encoded", "z_sample"),
) -> Model:
    """
    Gets the encoding layer from a model.
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


def _get_npz(path: str) -> np.array:
    """
    Loads all traces
    """
    _data = np.load
    X = _data(path, allow_pickle=True)["data"]
    return X


def _get_train_test_info(path: str) -> Tuple[np.array, np.array, float, float]:
    """
    Load indices for train/test split on the given dataset from a model dir.
    """
    f = np.load(os.path.join(path, "info.npz"), allow_pickle=True)
    idx_train, idx_test, mu, sg = f["info"]
    return idx_train, idx_test, mu, sg


@st.cache
def _standardize_train_test(X, mu, sigma, idx_train=None, idx_test=None):
    """
    Uses the indices to find the right train/test samples, and normalizes them.
    """
    if idx_train is None or idx_test is None:
        X_train, X_test = X[idx_train], X[idx_test]
        X = [lib.math.standardize(X, mu, sigma) for X in (X_train, X_test)]
    else:
        X = lib.math.standardize(X, mu, sigma)
    return X


def _get_latest_model(MODEL_DIR, recency=1):
    """
    Fetches latest model in directory.
    """
    models = glob(os.path.join(MODEL_DIR, "model*"))
    try:
        latest = sorted(models)[-recency]
        return latest
    except IndexError:
        st.write("Index error. Does the directory actually contain models?")


def _get_predictions(
    X_true, idx_true, model_path, savename, single_feature=None
):
    """
    Predicts autoencoder features and saves them. Saving as npz is required for
    interactivity, as predictions use the GPU (which may not be available while
    training). Warning: order is not preserved due to batching!
    """
    # See if there's a cached version
    try:
        f = np.load(
            "results/extracted_features/{}".format(savename), allow_pickle=True
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
        latest_model_path = _get_latest_model(model_path)

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
            "results/extracted_features/{}".format(savename),
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
def _bic(features, max_clusters = 20, spacing = 2):
    """
    Calculates AIC/BIC for different numbers of clusters.
    """
    rng = np.arange(1, max_clusters, spacing)
    bics, aics = [], []
    for i in rng:
        clf = sklearn.mixture.GaussianMixture(
            n_components=i, covariance_type="full"
        )
        clf.fit(features)
        bics.append(clf.bic(features))
        aics.append(clf.aic(features))
    return bics, aics, rng


@st.cache
def _pca(features, embed_into_n_components):
    """
    Calculates the PCA of raw input features.
    """
    pca = sklearn.decomposition.PCA(n_components=embed_into_n_components)
    pca_features = pca.fit_transform(features)
    explained_var = np.cumsum(np.round(pca.explained_variance_ratio_, 3))
    return pca_features, explained_var


@st.cache
def _manifold_embedding(features, embed_into_n_components):
    """
    Calculates the UMAP embedding of raw input features.
    """
    umap = UMAP(
        n_neighbors=50, min_dist=0.0, n_components=embed_into_n_components
    )
    manifold = umap.fit_transform(features)
    return manifold


@st.cache
def _cluster(features, n_clusters):
    """Performs clustering and PCA for visualization"""
    # cluster the decomposed
    clf = sklearn.mixture.GaussianMixture(
        n_components=n_clusters, covariance_type="full"
    )
    clf.fit(features)
    probs = clf.predict_proba(features)

    # categorical label
    labels = np.argmax(probs, axis=1)

    # confidence in predicted label
    top_prop = probs.max(axis=1)
    return labels, top_prop


@lib.utils.timeit
def _calculate_resampled_mean_std(traces, length, normalize_to_one):
    """
    Resamples all input traces to a given length and calculates mean/std.
    """
    # Resample first
    traces_re = parmap.map(
        lib.math.resample_timeseries, traces, length, pm_processes=16
    )

    # Normalize to 1
    if normalize_to_one:
        traces_re = [t / t.max(axis=0) for t in traces_re]

    trace_mean = np.mean(traces_re, axis=0)
    trace_err = np.std(traces_re, axis=0) / np.sqrt(len(traces_re))
    if len(trace_mean.shape) == 1:
        trace_mean = trace_mean.reshape(-1, 1)
        trace_err = trace_err.reshape(-1, 1)
    return trace_mean, trace_err


def _plot_bic(bic, aic, rng):
    """
    Plots BIC and AIC from Gaussian Mixture Model clustering.
    """
    fig, ax = plt.subplots()
    ax.plot(rng, aic, "o-", label="AIC", color="blue", ls=":")
    ax.plot(rng, bic, "o-", label="BIC", color="orange", ls=":")

    ax.axvline(rng[np.argmin(aic)], color="blue")
    ax.axvline(rng[np.argmin(bic)], color="orange")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(loc="upper right")
    ax.set_ylabel("Score")
    ax.set_xlabel("Number of clusters")

    plt.tight_layout()
    st.write(fig)


def _plot_explained_variance(explained_variance):
    """
    Plots explained PCA variance.
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


def _plot_embedding(embedding, cluster_labels):
    """
    Plots lower dimensional embedding with predicted cluster labels.
    """
    fig, ax = plt.subplots()

    lmin = min(cluster_labels)
    lmax = max(cluster_labels)

    embedding, cluster_labels = sklearn.utils.resample(
        embedding, cluster_labels, n_samples=2000
    )

    mids = []
    for i in range(len(set(cluster_labels))):
        emb_i = embedding[cluster_labels == i]
        mids.append(np.mean(emb_i, axis=0))

    cmap = plt.get_cmap("magma", lmax - lmin + 1)
    c1 = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cluster_labels,
        cmap=cmap,
        s=10,
        alpha=0.8,
    )
    ax.set_xlabel("C0")
    ax.set_ylabel("C1")
    fig.colorbar(c1, ax=ax, ticks=np.arange(lmin, lmax + 1))

    for i, m in enumerate(mids):
        ax.annotate(
            xy=m[[0, 1]],
            s=i,
            bbox=dict(boxstyle="square", fc="w", ec="grey", alpha=0.9),
        )

    plt.tight_layout()
    st.write(fig)


def _plot_adjustables(
    original_mse,
    original_confidence,
    mse_filter,
    confidence_filter,
    n_clusters,
):
    """
    Plots adjustables (MSE and confidence thresholds), given the original MSE
    and confidences (before filtering applied).
    """
    fig, ax = plt.subplots(nrows=2)
    ax[0].hist(
        original_mse,
        color="lightgrey",
        bins=np.arange(0, max(original_mse), 0.01),
        edgecolor="darkgrey",
        density=True,
    )
    ax[0].axvline(mse_filter, color="black", ls="--")
    ax[0].set_xlim(0, np.quantile(original_mse, 0.95))
    ax[0].set_xlabel("MSE")
    ax[0].set_ylabel("Probability density")

    confidence = []
    for i in range(n_clusters):
        y, _ = np.histogram(original_confidence, density=True)
        confidence.append(y)
    confidence = np.array(confidence)

    mean_conf = confidence.mean(axis=0)
    rng = np.linspace(0, 1, len(mean_conf))
    ax[1].plot(rng, mean_conf, "ko-")
    ax[1].set_xlabel("Label confidence")
    ax[1].set_ylabel("Probability density")
    ax[1].axvline(confidence_filter, color="black", ls="--")

    plt.tight_layout()
    svg_write(fig)


def _plot_traces_preview(
    X_true,
    X_pred,
    mse,
    sample_indices,
    confidence,
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
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        xi_true = X_true[i]
        xi_pred = X_pred[i]
        ci = confidence[i]
        ei = mse[i]
        idx = sample_indices[i]

        rng = range(len(xi_true))

        ax.set_title(
            "E = {:.2f}, L = {}, p={:.2f}\n"
            "idx = {}".format(ei, len(xi_true), ci, idx)
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


def _find_peaks(X, n_frames=3, n_std=2):
    """
    Finds peaks for input list of arrays.
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


def _plot_mean_trace(
    X, cluster_labels, cluster_lengths, percentages, n_rows_cols, colors
):
    """
    Plots mean and std of resampled traces for each cluster.
    """
    fig, axes = plt.subplots(nrows=n_rows_cols, ncols=n_rows_cols)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        try:
            (label_idx,) = np.where(cluster_labels == i)
            resample_len = np.max(cluster_lengths[i])
            trace_mean, trace_std = _calculate_resampled_mean_std(
                X[label_idx], length=resample_len, normalize_to_one=False
            )

            channels = trace_mean.shape[-1]
            for c in range(channels):
                cmean = trace_mean[..., c]
                cstd = trace_std[..., c]
                ax_ = ax.twinx()
                ax_.plot(cmean, color=colors[c])
                ax_.fill_between(
                    x=range(len(cmean)),
                    y1=cmean - cstd,
                    y2=cmean + cstd,
                    alpha=0.4,
                    color=colors[c],
                )
                ax_.set_yticks(())
                ax.set_yticks(())
            ax.set_title("{} ({:.1f} %)".format(i, percentages[i]))
        except IndexError:
            fig.delaxes(ax)
    plt.tight_layout()
    svg_write(fig)


def _plot_length_dist(cluster_lengths, n_rows_cols, colors):
    """
    Plots length distributions of each cluster.
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
    model_dir = sorted(glob("models/*"))
    dataset_dir = sorted(glob("results/intensities/*.npz"))

    model = st.selectbox(
        options=model_dir,
        index=len(model_dir) - 1,
        label="Select model",
        format_func=os.path.basename,
    )

    INTENSITIES = "results/intensities"
    default_dataset = re.search("data=.*.npz", model)[0].split("=")[1]

    default_dataset_idx = None
    for i, d in enumerate(dataset_dir):
        if os.path.basename(d) == default_dataset:
            default_dataset_idx = i
    if default_dataset is None:
        raise ValueError("Could not match dataset with any in directory")

    model_name = os.path.basename(model)
    selected_dataset = st.selectbox(
        label="Select dataset to predict on",
        options=dataset_dir,
        index=default_dataset_idx,
        format_func=os.path.basename,
    )

    st.write("**Model**:", model_name)
    st.write("**Dataset**:", selected_dataset)

    # Predict only on the test set (not really using the training set for
    # anything downstream at the moment
    single_feature = re.search("single=.*", model)
    if single_feature is not None:
        print("Using only single channel")
        single_feature = int(single_feature[0][-1])

    # Always load mu and sg obtained from train set
    idx_train, idx_test, mu_train, sg_train = _get_train_test_info(model)

    if selected_dataset == default_dataset:
        use_data = st.sidebar.radio(
            label="Dataset to use for predictions",
            options=["test", "train", "combine"],
            index=0,
        )

        # fitted traces for autoencoder predictions
        X = _get_npz(os.path.join(INTENSITIES, selected_dataset))

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
        elif use_data == "combine":
            X_true = np.concatenate((X_train, X_test))
            idx_true = np.concatenate((idx_train, idx_test))
        else:
            raise ValueError("Invalid data selection")
    else:
        st.subheader("**External dataset chosen**")
        # Take a completely different dataset
        X_true = np.load(selected_dataset, allow_pickle=True)["data"]
        X_true = lib.math.standardize(X_true, mu=mu_train, sigma=sg_train)
        idx_true = np.array(range(len(X_true)))

    len_X_true_prefilter = len(X_true)

    X_true, X_pred, features, mse, sample_indices = _get_predictions(
        X_true=X_true,
        idx_true=idx_true,
        model_path=model,
        single_feature=single_feature,
        savename=model_name + "___pred_" + os.path.basename(selected_dataset),
    )
    mse_orig = mse.copy()
    # sample_indices_orig = sample_indices.copy()

    StWidgets = _setup_streamlit_widgets(
        n_components_max=features.shape[-1], original_mse=mse_orig
    )

    # Keep only datapoints with len above a minimum length
    arr_lens = np.array([len(xi) for xi in X_true])
    (len_above_idx,) = np.where(arr_lens >= StWidgets.min_length)
    X_true, X_pred, features, mse, sample_indices = get_index(
        X_true, X_pred, features, mse, sample_indices, index=len_above_idx
    )

    # Keep only datapoints with error below threshold and cluster these
    # (high error data shouldn't be there)
    (errorbelow_idx,) = np.where(mse < StWidgets.mse_filter)
    X_true, X_pred, features, mse, sample_indices = get_index(
        X_true, X_pred, features, mse, sample_indices, index=errorbelow_idx
    )
    # perform both PCA and manifold of features
    pca, pca_variance = _pca(
        features, embed_into_n_components=StWidgets.n_components
    )
    pca_orig = pca.copy()
    # umap = _manifold_embedding(features, embed_into_n_components = 5)

    if StWidgets.feature_type == "raw":
        modified_features = features
    elif StWidgets.feature_type == "pca":
        modified_features = pca
    else:
        raise ValueError("Invalid feature type")

    cluster_labels, label_confidence = _cluster(
        features=modified_features, n_clusters=StWidgets.n_clusters,
    )

    cluster_labels_orig = cluster_labels.copy()
    label_confidence_orig = label_confidence.copy()

    # Keep only datapoints with confidence above threshold
    (confidence_idx,) = np.where(label_confidence > StWidgets.confidence_filter)

    (
        X_true,
        X_pred,
        modified_features,
        mse,
        label_confidence,
        cluster_labels,
        pca,
        sample_indices,
    ) = get_index(
        X_true,
        X_pred,
        modified_features,
        mse,
        label_confidence,
        cluster_labels,
        pca,
        sample_indices,
        index=confidence_idx,
    )

    len_X_true_postfilter = len(X_true)
    n_rows_cols = int(np.ceil(np.sqrt(StWidgets.n_clusters)))
    colormap = lib.plotting.get_colors("viridis", n_colors=StWidgets.n_clusters)
    linemap = lib.plotting.get_colors("inferno", n_colors=X_true[0].shape[-1])

    st.subheader("Total number of traces in dataset")
    st.write(
        "Pre-filter, N = {}\n"
        "Post-filter, N = {}\n"
        "Removed fraction: {:.2f}".format(
            len_X_true_prefilter,
            len_X_true_postfilter,
            1 - len_X_true_postfilter / len_X_true_prefilter,
        )
    )

    st.subheader("Clustering of features")
    _plot_adjustables(
        original_mse=mse_orig,
        confidence_filter=StWidgets.confidence_filter,
        mse_filter=StWidgets.mse_filter,
        original_confidence=label_confidence_orig,
        n_clusters=StWidgets.n_clusters,
    )

    st.subheader("PCA of traces, only filtered by MSE")
    _plot_embedding(embedding=pca_orig, cluster_labels=cluster_labels_orig)
    _plot_explained_variance(pca_variance)

    st.subheader("PCA of final selection of traces")
    _plot_embedding(embedding=pca, cluster_labels=cluster_labels)

    if st.sidebar.checkbox(label="Run BIC", value=False):
        st.subheader("Most likely number of clusters")
        bic, aic, rng = _bic(modified_features)
        _plot_bic(bic, aic, rng)

    lengths_in_cluster = []
    percentage_of_samples = []
    for i in range(StWidgets.n_clusters):
        (idx,) = np.where(cluster_labels == i)
        percentage = (len(idx) / len_X_true_postfilter) * 100

        percentage_of_samples.append(percentage)
        if percentage > 1:
            len_i = [len(xi) for xi in X_true[idx]]

            mse_i = mse[idx]
            conf_i = label_confidence[idx]
            X_true_i = X_true[idx]
            X_pred_i = X_pred[idx]
            sample_indices_i = sample_indices[idx]
            lengths_in_cluster.append(len_i)

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

            _plot_traces_preview(
                X_true=X_true_i,
                X_pred=X_pred_i,
                confidence=conf_i,
                sample_indices=sample_indices_i,
                mse=mse_i,
                nrows=StWidgets.nrows,
                ncols=StWidgets.ncols,
                plot_real_values=StWidgets.plot_real_values,
                separate_y_ax=StWidgets.separate_y,
                mu=mu_train,
                sg=sg_train,
                single_feature=single_feature,
                colors=linemap,
            )

        else:
            st.subheader(
                "Cluster {} contains less than 1% of samples!".format(i)
            )
            continue

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
        colors=linemap,
        percentages=percentage_of_samples,
    )

    # st.subheader("Mean (resampled) PREDICTED traces in each cluster")
    # _plot_mean_trace(
    #     X = X_pred,
    #     cluster_labels = cluster_labels,
    #     cluster_lengths = cluster_lengths,
    #     n_rows_cols = n_rows_cols,
    #     colors = linemap
    # )

    # st.subheader("Intensity distribution per trace")
    # _plot_max_intensity(
    #     X=X_true,
    #     cluster_labels=cluster_labels,
    #     n_rows_cols=n_rows_cols,
    #     mu=mu_train,
    #     sg=sg_train,
    #     colors=colormap,
    # )


if __name__ == "__main__":
    main()
