import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.decomposition
import sklearn.mixture
import sklearn.neighbors
import sklearn.preprocessing
import streamlit as st
import sklearn.cluster
import lib.math

np.random.seed(1)


def color_generator(n_colors):
    return iter(plt.cm.rainbow(np.linspace(0, 1, n_colors)))


def make_ellipses(gmm, ax):
    k_components = len(gmm.means_)
    cgen = color_generator(k_components)

    for n, color in enumerate(cgen):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        else:
            raise ValueError
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = matplotlib.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], 180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        # ax.set_aspect("equal", "datalim")


@st.cache
def local_outliers(X, cutoff = 3.5):
    """Detects local outliers and plots them"""
    clf = sklearn.neighbors.LocalOutlierFactor(
        n_neighbors=20, contamination=0.1
    )
    clf.fit_predict(X)

    scores = clf.negative_outlier_factor_
    z_score = lib.math.modified_z_score(scores)
    (inlier_idx,) = np.where(z_score <= cutoff)
    (outlier_idx,) = np.where(z_score > cutoff)

    return z_score, scores, inlier_idx, outlier_idx, cutoff


@st.cache
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
    f = np.load(
        "results/extracted_features/20191202-1542_lstm_autoencoder_dim=64_activation=elu_variable_data=tracks-CLTA-TagRFP EGFP-Gak-A8_var_predictions.npz"
    )
    features = f["features"]
    features = sklearn.utils.resample(features, n_samples=300)

    st.subheader("Outlier removal")
    X = sklearn.preprocessing.scale(features)

    (
        mean_abs_dev,
        outlier_score,
        inlier_idx,
        outlier_idx,
        cutoff_val,
    ) = local_outliers(X)

    outliers = X[outlier_idx]

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.ravel()
    ax[0].set_title("Local Outlier Factor")
    ax[0].scatter(X[:, 0], X[:, 1], s=5, label="Data points")

    radius = (outlier_score.max() - outlier_score) / (
        outlier_score.max() - outlier_score.min()
    )
    ax[0].scatter(
        X[:, 0],
        X[:, 1],
        s=1000 * radius,
        edgecolors="r",
        facecolors="none",
        label="Outlier scores",
    )

    ax[1].set_title("Outlier score after filtering")
    ax[1].hist(outlier_score, bins=30)
    ax[1].set_yscale("log")

    ax[2].set_title("MAD")
    ax[2].hist(mean_abs_dev)
    ax[2].axvline(cutoff_val, color="black")

    ax[3].set_title("Outliers in red")
    ax[3].scatter(X[:, 0], X[:, 1], s=5, color="black")
    ax[3].scatter(outliers[:, 0], outliers[:, 1], color="red", s=5)
    plt.tight_layout()

    st.write(fig)

    X_filtered = X[inlier_idx]

    st.subheader("PCA after outlier removal")
    pca = sklearn.decomposition.KernelPCA(n_components = 32, kernel = "poly", degree = 2)
    X_pca_unfiltered = pca.fit_transform(X)
    X_pca = pca.fit_transform(X_filtered)
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title("Unfiltered PCA")
    ax[0].scatter(X_pca_unfiltered[:, 0], X_pca_unfiltered[:, 1])

    ax[1].set_title("Filtered PCA")
    ax[1].scatter(X_pca[:, 0], X_pca[:, 1])
    st.write(fig)

    gmm, params, bics, best_k, k = lib.math.fit_gaussian_mixture(
        X_pca, k_min = 1, k_max = 40, step_size = 2
    )
    labels = gmm.predict(X_pca)

    fig, ax = plt.subplots(ncols=3, figsize=(7, 3))
    ax = ax.ravel()

    st.subheader("Gaussian mixture model")
    ax[0].set_title("GMM clusters")
    ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c = labels, edgecolor= "grey")

    ax[1].set_title("BIC")
    ax[1].plot(k, bics)
    ax[1].axvline(k[np.argmin(bics)])

    ax[2].set_title("Number of items per cluster")
    ax[2].hist(labels, bins = np.arange(0, best_k+1, 1))

    plt.tight_layout()
    st.write(fig)
