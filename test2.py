import os.path
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import mpl_scatter_density
import numpy as np
import sklearn.utils
import umap.umap_ as umap
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

import lib.globals
import lib.globals
from lib.math import resample_timeseries


def _plot_kmeans_scores(X, min, max, step):
    """
    Calculates scores for multiple values of kmeans
    Args:
        X (np.ndarray)
        min (int)
        max (int)
        step (int)
    """
    rng = list(range(min, max, step))

    def process(n):
        clf = GaussianMixture(n_components = n, random_state = 42)
        # clf = MiniBatchKMeans(n_clusters=n, random_state=42)
        labels = clf.fit_predict(X)

        s = silhouette_score(X, labels)
        c = calinski_harabasz_score(X, labels)
        b = clf.bic(X)

        return s, c, b

    n_jobs = len(rng)
    results = Parallel(n_jobs=n_jobs)(delayed(process)(i) for i in tqdm(rng))
    results = np.column_stack(results).T

    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(rng, results[:, 0], "o-", color="blue", label="Silhouette score")
    ax[1].plot(rng, results[:, 1], "o-", color="orange", label="CH score")
    ax[2].plot(rng, results[:, 2], "o-", color="red", label="BIC")

    for a in ax:
        a.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("plots/best_k.pdf")
    plt.show()


def main(encodings_name):
    np.random.seed(1)

    f = np.load(
        os.path.join(lib.globals.encodings_dir, encodings_name),
        allow_pickle=True,
    )

    X = f["X_true"]
    X = sklearn.utils.resample(X, n_samples = 10000, replace = False)

    X = np.array([lib.math.resample_timeseries(xi.ravel().reshape(-1, 1), new_length = 200) for xi in X])
    X = np.squeeze(X)

    u = umap.UMAP(
        n_components = 2,
        random_state = 42,
        n_neighbors = 100,
        min_dist = 0.0,
        init = "spectral",
        verbose = True,
    )
    e = u.fit_transform(X)

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection = "scatter_density")

    clf = MiniBatchKMeans(n_clusters = 35)
    labels = clf.fit_predict(e)
    centers = clf.cluster_centers_

    ax.scatter_density(e[:, 0], e[:, 1], c = labels, cmap = "viridis_r")

    for i in range(len(set(labels))):
        m = centers[i]
        ax.annotate(
            xy = m,
            s = i,
            bbox = dict(boxstyle = "square", fc = "w", ec = "grey", alpha = 0.9),
        )

    Xa = X[labels == 26]
    Xb = X[labels == 33]

    colors = ["black", "red"]

    for X in Xb, Xa:
        fig, ax = plt.subplots(nrows = 5, ncols = 5)
        ax = ax.ravel()
        for i in range(len(ax)):
            xi = np.zeros((100, 2))
            xi[:, 0] = X[i][:100]
            xi[:, 1] = X[i][100:]

            for c in range(xi.shape[-1]):
                ax[i].plot(xi[..., c], color = colors[c])

            ax[i].set_xticks(())
            ax[i].set_yticks(())
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    NAME = "20200124-0206_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=0.1_zdim=8_anneal=20___pred__combined_filt20_var.npz"
    main(NAME)
