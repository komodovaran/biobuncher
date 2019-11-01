import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import lib.utils
import tsfresh.feature_extraction.feature_calculators as fc

from lib.plotting import _plot_c0_c1


@st.cache
def _get_data():
    """
    Loads all traces and converts them to a padded tensor
    """
    df = pd.read_hdf("results/2_intensities/2_intensities.h5")  # type: pd.DataFrame
    # # TODO: remove this line when done
    # df = lib.utils.sample_groups(df, size=400, by=["file", "particle"])
    return df


def _process(df):
    """
    Saves filtered dataframe to NumPy array
    """
    len_per_group = df.groupby(["file", "particle"]).apply(lambda x: len(x))
    max_len = np.max(len_per_group)
    n_groups = len(len_per_group)
    columns = ["int_c0", "int_c1", "steplength"]

    X = np.zeros(shape=(n_groups, max_len, len(columns)))
    for n, (_, group) in enumerate(df.groupby(["file", "particle"])):
        pad = max_len - len(group)
        X[n, pad:, 0] = group["int_c0"]
        X[n, pad:, 1] = group["int_c1"]
        X[n, pad:, 2] = group["steplength"]

    np.savez(file="results/2_intensities/2_intensities_padded_filtered.npz")


if __name__ == "__main__":
    by = ["file", "particle"]
    df = _get_data()
    st.write(df.head())
    st.write(df.columns)

    grouped_df = df.groupby(by)

    rel_change = grouped_df.apply(lambda x: x["int_c1"].max() / x["int_c1"].min())
    lengths = grouped_df.apply(lambda x: len(x))

    st.subheader("Relative change C1 of groups")
    fig, ax = plt.subplots()
    ax.hist(rel_change.values, bins=30)
    st.write(fig)

    st.subheader("Relative change C1 vs length")
    fig, ax = plt.subplots()
    ax.scatter(rel_change, lengths)
    ax.set_ylabel("Length of group")
    ax.set_xlabel("relative change C1")
    st.write(fig)

    st.subheader("Anything with a relative change <2 and no peaks in C1 (Aux) is useless data")
    samples = lib.utils.sample_groups(df, size=16, by=by)
    fig, ax = plt.subplots(nrows=4, ncols=4)
    ax = ax.ravel()
    for n, (_, g) in enumerate(samples.groupby(by)):
        _plot_c0_c1(ax=ax[n], int_c0=g["int_c0"], int_c1=g["int_c1"])

        n_peaks = fc.number_peaks(g["int_c1"].values, n=7)

        ax[n].set_title(
            "rel = {:.1f}\npeaks = {}".format(g["int_c1"].max() / g["int_c1"].min(), n_peaks)
        )
    plt.tight_layout()
    st.write(fig)

    def _filter(g):  # ~ 2.5 ms per group
        n_peaks = fc.number_peaks(g["int_c1"], n=5)
        relative_change = g["int_c1"].max() / g["int_c1"].min()

        condition_1 = len(g) > 15
        condition_2 = 1 <= n_peaks <= 9
        condition_3 = relative_change > 2

        if all((condition_1, condition_2, condition_3)):
            return g
        else:
            return None

    filtered_df = lib.utils.groupby_parallel_apply(grouped_df, _filter, concat=True)
    st.write(len(filtered_df.groupby(by)))

    st.subheader("After filtering:")
    samples = lib.utils.sample_groups(filtered_df, size=16, by=by)
    fig, ax = plt.subplots(nrows=4, ncols=4)
    ax = ax.ravel()
    for n, (_, g) in enumerate(samples.groupby(by)):
        _plot_c0_c1(ax=ax[n], int_c0=g["int_c0"], int_c1=g["int_c1"])
        n_peaks = fc.number_peaks(g["int_c1"].values, n=5)
        ax[n].set_title(
            "rel = {:.1f}\npeaks = {}".format(g["int_c1"].max() / g["int_c1"].min(), n_peaks)
        )
    plt.tight_layout()
    st.write(fig)

    _process(filtered_df)
