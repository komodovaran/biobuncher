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
def _get_data(path):
    """
    Loads all traces and converts them to a padded tensor
    """
    df = pd.DataFrame(pd.read_hdf(path))
    # df = lib.utils.sample_groups(df, size=400, by=["file", "particle"]) # for testing only
    return df


def _process(df, path, by):
    """
    Saves filtered dataframe to NumPy array
    """
    grouped_df = df.groupby(by)

    len_per_group = grouped_df.apply(lambda x: len(x))
    median_len = np.median(len_per_group)

    n_groups = len(len_per_group)
    columns = ["int_c0", "int_c1"]

    X_resampled = np.zeros(shape = (n_groups, int(median_len), len(columns)))
    for n, (_, group) in enumerate(grouped_df):
        features = group[["int_c0", "int_c1"]].values

        features_resampled = [lib.math.resample_timeseries(y, new_length = X_resampled.shape[1]) for y in features.T]
        features_resampled = np.column_stack(features_resampled)
        X_resampled[n, ...] = features_resampled

    np.savez(path[:-3] + "_resampled-medianlen.npz", data = X_resampled)
    return X_resampled

if __name__ == "__main__":
    INPUT_DIR = "results/intensities/tracks-tpy.h5"
    BY = ["file", "particle", "split"]

    df = _get_data(INPUT_DIR)
    st.write(df.head())
    st.write(df.columns)

    grouped_df = df.groupby(BY)

    def _filter(group):
        return group if len(group) > 30 else None

    filtered_df = lib.utils.groupby_parallel_apply(grouped_df, _filter, concat = True)
    # print("Filtering disabled")
    st.write(len(filtered_df.groupby(BY)))

    st.subheader("After filtering")
    samples = lib.utils.sample_groups(filtered_df, size = 16, by = BY)
    fig, ax = plt.subplots(nrows = 4, ncols = 4)
    ax = ax.ravel()
    for n, (_, g) in enumerate(samples.groupby(BY)):
        plot_c0_c1(ax = ax[n], int_c0 = g["int_c0"], int_c1 = g["int_c1"])
        n_peaks = fc.number_peaks(g["int_c1"].values, n = 5)
        ax[n].set_title(
            "rel = {:.1f}\npeaks = {}".format(g["int_c1"].max() / g["int_c1"].min(), n_peaks)
        )
    plt.tight_layout()
    st.write(fig)

    X_resampled = _process(filtered_df, path = INPUT_DIR, by = BY)

    st.subheader("Examples after resampling")
    fig, ax = plt.subplots(nrows =2, ncols = 2)
    ax = ax.ravel()
    for n in range(len(ax)):
        xi = X_resampled[n]
        ax[n].plot(xi)
    plt.tight_layout()
    st.write(fig)