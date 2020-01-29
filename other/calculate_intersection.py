import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lib.utils
import sklearn.model_selection
import lib.math


def calculate_iou_matrix(dict_a, dict_b):
    """
    Calculates the intersection over union for two dictionaries of sets

    Args:
          dict_a (dict)
          dict_b (dict)
    """

    z = []
    for ki, vi in dict_a.items():
        for kj, vj in dict_b.items():
            intersection = len(vi.intersection(vj))
            union = len(vi.union(vj))
            iou = intersection / union
            z.append(iou)
    sqrt_l = int(np.sqrt(len(z)))
    z = np.reshape(z, (sqrt_l, sqrt_l))
    return z

def chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

if __name__ == "__main__":
    # single = "../results/cluster_indices/20200123-2208_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=0.1_zdim=8_anneal=20_single=0___clust_50__combined_filt5_var.npz__cidx.h5"
    # double = "../results/cluster_indices/20200124-0206_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=0.1_zdim=8_anneal=20___clust_50__combined_filt5_var.npz__cidx.h5"
    # single, double = [pd.read_hdf(p) for p in (single, double)]
    #
    # dicts = []
    # for data in single, double:
    #     d = {}
    #     for cluster in list(data["cluster"].unique()):
    #         d[cluster] = set(data["id"][data["cluster"] == cluster].values)
    #
    #     dicts.append(d)
    #


    idx_a = np.arange(0, 50000, 1)
    idx_b = np.arange(0, 50000, 1)
    np.random.shuffle(idx_a)
    np.random.shuffle(idx_b)

    idx_a = chunk(idx_a, 50)
    idx_b = chunk(idx_b, 50)

    dicts = []
    for data in idx_a, idx_b:
        d = {}
        for i in range(50):
            d[i] = set(data[i])
        dicts.append(d)


    single, double = dicts
    m = calculate_iou_matrix(single, double)
    c = plt.matshow(m)
    # plt.xlabel("Double")
    # plt.ylabel("Single")
    plt.xlabel("Random 1")
    plt.xlabel("Random 2")

    cbar = plt.colorbar(c)
    plt.clim(0, 0.5)

    plt.savefig("../plots/iou_random_50.pdf")
    plt.show()