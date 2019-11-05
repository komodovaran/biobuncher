import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lib.utils
import lib.plotting
import lib.math
import streamlit as st

@st.cache
def _get_data(include_steplength=False):
    """
    Loads all traces and converts them to a padded tensor
    """
    df = pd.DataFrame(pd.read_hdf("results/intensities/2_intensities.h5"))
    df = df.groupby(["file", "particle"]).filter(lambda x: len(x) > 50)

    len_per_group = df.groupby(["file", "particle"]).apply(lambda x: len(x))
    max_len = np.max(len_per_group)
    n_groups = len(len_per_group)
    n_channels = 2

    columns = ["int_c0", "int_c1", "steplength"]
    if not include_steplength:
        columns.pop(-1)


    X = np.zeros(shape=(n_groups, max_len, n_channels))
    for n, (_, group) in enumerate(df.groupby(["file", "particle"])):
        pad = max_len - len(group)
        X[n, pad:, 0] = group["int_c0"]
        X[n, pad:, 1] = group["int_c1"]
    return X

def _plot_examples(X_raw):
    nrows = 5
    ncols = 3

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize = (6, 5))
    ax = ax.ravel()
    rand_idx = np.random.randint(0, len(X), nrows * ncols).tolist()

    for i, r in enumerate(rand_idx):
        xi, = lib.utils.remove_zero_padding(X_raw[r, ...])
        ax[i].plot(xi[:, 0], color="salmon")
        ax[i].plot(xi[:, 1], color="lightgreen")
        ax[i].set_xticks(())
        ax[i].set_yticks(())
    fig.legend(
        lib.plotting.create_legend_handles(("salmon", "lightgreen")),
        ["TagRFP", "EGFP"],
        loc="upper right",
        framealpha=1,
    )
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    X_raw = _get_data()
    X = lib.math.normalize_tensor(X_raw)
    st.write(_plot_examples(X_raw))