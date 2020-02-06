import os.path
import os.path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from kneed import KneeLocator
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

import lib.globals
import lib.math
import lib.utils
from lib.utils import get_index, random_subset


def _locate_knee(x, y, direction="decreasing"):
    kn = KneeLocator(
        x, y, curve="convex", direction=direction, interp_method="interp1d",
    )
    return kn.knee


def _plot_scores(X, min, max, step):
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
        clf = GaussianMixture(n_components=n, random_state=42)
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

    results = []
    for proc in process_kmeans, process_gaussian:
        r = Parallel(n_jobs=n_jobs)(delayed(proc)(i) for i in tqdm(rng))
        r = np.column_stack(r).T
        results.append(r)

    r_kmeans, r_gmm = results

    fig, ax = plt.subplots(nrows=3, figsize=(10, 10))
    ax = ax.ravel()

    k0, k1 = [_locate_knee(rng, r) for r in (r_kmeans[:, 0], r_gmm[:, 0])]
    ax[0].set_title("Silhouette score (higher is better)")
    ax[0].plot(
        rng,
        r_kmeans[:, 0] / r_kmeans[:, 0].max(),
        "o-",
        color="blue",
        label="K-means",
    )
    ax[0].plot(
        rng,
        r_gmm[:, 0] / r_gmm[:, 0].max(),
        "o-",
        color="orange",
        label="Gaussian mixture",
    )

    ax[0].axvline(k0, color="blue", label="best K-means ({})".format(k0))
    ax[0].axvline(
        k1, color="orange", label="best Gaussian mixture ({})".format(k1)
    )

    k0, k1 = [_locate_knee(rng, r) for r in (r_kmeans[:, 1], r_gmm[:, 1])]
    ax[1].set_title("CH score (lower is better)")
    ax[1].plot(
        rng,
        r_kmeans[:, 1] / r_kmeans[:, 1].max(),
        "o-",
        color="orange",
        label="K-means",
    )
    ax[1].axvline(k0, color="blue", label="best K-means ({})".format(k0))
    ax[1].plot(
        rng,
        r_gmm[:, 1] / r_gmm[:, 1].max(),
        "o-",
        color="blue",
        label="Gaussian mixture",
    )
    ax[1].axvline(
        k1, color="orange", label="best Gaussian mixture ({})".format(k1)
    )

    k_bic = _locate_knee(rng, r_gmm[:, 2])
    ax[2].set_title("BIC")
    ax[2].plot(
        rng, r_gmm[:, 2], "o-", color="red", label="BIC (lower is better)"
    )
    ax[2].axvline(
        k_bic,
        color="red",
        ls="--",
        label="best Gaussian mixture ({})".format(k_bic),
    )

    for a in ax.ravel():
        a.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("plots/best_clusters.pdf")
    plt.show()


def main(
    encodings_name, min_track_len, cmin, cmax, cstep, subset_n_samples=None
):
    f = np.load(
        os.path.join(lib.globals.encodings_dir, encodings_name),
        allow_pickle=True,
    )

    X, encodings = f["X_true"], f["features"]
    if subset_n_samples is not None:
        X, encodings = random_subset((X, encodings), n_samples=subset_n_samples)

    arr_lens = np.array([len(xi) for xi in X])
    (len_above_idx,) = np.where(arr_lens >= min_track_len)
    X, encodings, = get_index((X, encodings), index=len_above_idx)

    _plot_scores(encodings, min=cmin, max=cmax, step=cstep)


if __name__ == "__main__":
    NAME = "20200201-1107_lstm_vae_bidir_data=combined_filt20_var.npz_dim=128_act=None_bat=4_eps=1_zdim=16_anneal=5___pred__combined_filt20_var.npz"
    main(
        encodings_name = NAME,
        min_track_len=20,
        cmin=2,
        cmax=40,
        cstep=2,
        subset_n_samples=100000,
    )
