import os.path
import os.path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

import lib.globals
import lib.globals
from lib.utils import get_index


def _plot_kmeans_score(X, min, max, step):
    """
    Calculates scores for multiple values of kmeans
    Args:
        X (np.ndarray)
        min (int)
        max (int)
        step (int)
    """
    rng = list(range(min, max, step))

    def process_gaussian(n):
        clf = GaussianMixture(n_components = n, random_state = 42)
        labels = clf.fit_predict(X)

        s = silhouette_score(X, labels)
        c = calinski_harabasz_score(X, labels)
        b = clf.bic(X)

        return s, c, b

    def process_kmeans(n):
        clf = MiniBatchKMeans(n_clusters=n, random_state=42)
        labels = clf.fit_predict(X)

        s = silhouette_score(X, labels)
        c = calinski_harabasz_score(X, labels)
        return s, c


    n_jobs = len(rng)
    results_kmeans = Parallel(n_jobs=n_jobs)(delayed(process_kmeans)(i) for i in tqdm(rng))
    results_kmeans = np.column_stack(results_kmeans).T

    fig, ax = plt.subplots(nrows=3, ncols = 2)

    ax[0, 0].set_title("K-means")
    ax[0, 0].plot(rng, results_kmeans[:, 0], "o-", color="blue", label="Silhouette score")
    ax[1, 0].plot(rng, results_kmeans[:, 1], "o-", color="orange", label="CH score")

    ax[0, 1].set_title("Gaussian Mixture")
    ax[0, 1].plot(rng, results_kmeans[:, 0], "o-", color="blue", label="Silhouette score")
    ax[1, 1].plot(rng, results_kmeans[:, 1], "o-", color="orange", label="CH score")
    ax[2, 1].plot(rng, results_kmeans[:, 2], "o-", color="red", label="BIC")

    for a in ax:
        a.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("plots/best_clusters.pdf")
    plt.show()


def main(encodings_name):
    f = np.load(
        os.path.join(lib.globals.encodings_dir, encodings_name),
        allow_pickle=True,
    )

    X, encodings = f["X_true"], f["features"]

    X = X[0:1000]
    encodings = encodings[0:1000]

    arr_lens = np.array([len(xi) for xi in X])
    (len_above_idx,) = np.where(arr_lens >= 30)
    X, encodings, = get_index((X, encodings), index=len_above_idx)

    _plot_kmeans_score(encodings, min=2, max=100, step=3)


if __name__ == "__main__":
    NAME = "20200124-0206_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=0.1_zdim=8_anneal=20___pred__combined_filt20_var.npz"
    main(NAME)
