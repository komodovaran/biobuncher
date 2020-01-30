import os.path
import os.path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from lib.utils import get_index

import lib.globals
import lib.globals


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
    f = np.load(
        os.path.join(lib.globals.encodings_dir, encodings_name),
        allow_pickle=True,
    )

    X, encodings = f["X_true"], f["features"]

    arr_lens = np.array([len(xi) for xi in X])
    (len_above_idx,) = np.where(arr_lens >= 30)
    X, encodings, = get_index((X, encodings), index=len_above_idx)

    _plot_kmeans_scores(encodings, min=2, max=100, step=3)


if __name__ == "__main__":
    NAME = "20200124-0206_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=0.1_zdim=8_anneal=20___pred__combined_filt20_var.npz"
    main(NAME)
