import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tsfresh.feature_extraction.feature_calculators as fc

import lib.math
import lib.utils
from lib.plotting import plot_c0_c1


@st.cache
def _get_data():
    """
    Loads all traces and converts them to a padded tensor
    """
    df = pd.read_hdf("results/intensities/intensities.h5")  # type: pd.DataFrame
    # df = lib.utils.sample_groups(df, size=400, by=["file", "particle"]) # for testing only
    return df


def _process(df):
    """
    Saves filtered dataframe to NumPy array
    """
    df["c0_c1_ratio"] = df["int_c0"] / (df["int_c0"] + df["int_c1"])
    df.replace(np.nan, 0, inplace = True)

    len_per_group = df.groupby(["file", "particle"]).apply(lambda x: len(x))
    max_len = np.max(len_per_group)
    n_groups = len(len_per_group)
    columns = ["int_c0", "int_c1", "c0_c1_ratio", "steplength"]

    X_padded = np.zeros(shape = (n_groups, max_len, len(columns)))
    X_resampled = np.zeros(shape = (n_groups, 50, len(columns)))

    for n, (_, group) in enumerate(df.groupby(["file", "particle"])):
        features = group[["int_c0", "int_c1", "c0_c1_ratio", "steplength"]].values

        # Option 1: Add extracted_features timeseries after zero-padding
        pad_len = max_len - len(group)
        X_padded[n, pad_len:, :] = features

        # Option 2: Resample all timeseries to be the same length (as max length)
        features_resampled = [lib.math.resample_timeseries(y, new_length = X_resampled.shape[1]) for y in features.T]
        features_resampled = np.column_stack(features_resampled)
        X_resampled[n, ...] = features_resampled

    np.savez("results/intensities/intensities_padded_minlen15.npz", data = X_padded)
    np.savez("results/intensities/intensities_resampled_minlen15_relen50.npz", data = X_resampled)

    return X_padded, X_resampled


if __name__ == "__main__":
    by = ["file", "particle"]
    df = _get_data()
    st.write(df.head())
    st.write(df.columns)

    grouped_df = df.groupby(by)

    rel_change = grouped_df.apply(lambda x: x["int_c0"].max() / x["int_c0"].min())
    lengths = grouped_df.apply(lambda x: len(x))

    st.subheader("Relative change C1 of groups")
    fig, ax = plt.subplots()
    ax.hist(rel_change.values, bins = 30)
    st.write(fig)

    st.subheader("Relative change C1 vs length")
    fig, ax = plt.subplots()
    ax.scatter(rel_change, lengths)
    ax.set_ylabel("Length of group")
    ax.set_xlabel("relative change C1")
    st.write(fig)

    st.subheader("Before filtering\nAnything with a relative change <2 in clathrin is useless data, so remove those")
    samples = lib.utils.sample_groups(df, size = 16, by = by)
    fig, ax = plt.subplots(nrows = 4, ncols = 4)
    ax = ax.ravel()
    for n, (_, g) in enumerate(samples.groupby(by)):
        plot_c0_c1(ax = ax[n], int_c0 = g["int_c0"], int_c1 = g["int_c1"])

        n_peaks = fc.number_peaks(g["int_c1"].values, n = 7)

        ax[n].set_title(
            "rel = {:.1f}\npeaks = {}".format(g["int_c1"].max() / g["int_c1"].min(), n_peaks)
        )
    plt.tight_layout()
    st.write(fig)

    # filtered_df = lib.utils.groupby_parallel_apply(grouped_df, _filter, concat = True)
    filtered_df = df
    print("Filtering disabled")
    st.write(len(filtered_df.groupby(by)))

    st.subheader("After filtering")
    samples = lib.utils.sample_groups(filtered_df, size = 16, by = by)
    fig, ax = plt.subplots(nrows = 4, ncols = 4)
    ax = ax.ravel()
    for n, (_, g) in enumerate(samples.groupby(by)):
        plot_c0_c1(ax = ax[n], int_c0 = g["int_c0"], int_c1 = g["int_c1"])
        n_peaks = fc.number_peaks(g["int_c1"].values, n = 5)
        ax[n].set_title(
            "rel = {:.1f}\npeaks = {}".format(g["int_c1"].max() / g["int_c1"].min(), n_peaks)
        )
    plt.tight_layout()
    st.write(fig)

    X_padded, X_upsampled = _process(filtered_df)

    st.subheader("Examples after padding (zeros removed)")
    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    ax = ax.ravel()
    for n in range(len(ax)):
        xi = X_padded[n]
        xi, = lib.utils.remove_zero_padding(xi)

        ax[n].plot(xi)
    plt.tight_layout()
    st.write(fig)

    st.subheader("Examples after resampling")
    fig, ax = plt.subplots(nrows =2, ncols = 2)
    ax = ax.ravel()
    for n in range(len(ax)):
        xi = X_upsampled[n]
        ax[n].plot(xi)
    plt.tight_layout()
    st.write(fig)