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
    return df


def _resample(group, length):
    """Resample with a length of 50 for all types of samples"""
    f = group[["int_c0", "int_c1"]].values
    f_resampled = [
        lib.math.resample_timeseries(y, new_length=length) for y in f.T
    ]
    return np.column_stack(f_resampled)


def _process(df, path, by):
    """
    Saves filtered dataframe to NumPy array
    """
    grouped_df = df.groupby(by)
    resample_length = 30
    print("median length: ", np.median(grouped_df.apply(len)))
    X_resampled = np.array(
        lib.utils.groupby_parallel_apply(
            func=_resample, grouped_df=grouped_df, f_args =(resample_length), concat=False, n_jobs = 16,
        )
    )
    np.savez(path[:-3] + "_resampled-{}.npz".format(resample_length), data=X_resampled)
    return X_resampled


if __name__ == "__main__":
    INPUT_DIR = "results/intensities/tracks-tpy_roi-int.h5"
    BY = ["file", "particle", "split"]
    FILTER_SHORT = 20

    df = _get_data(INPUT_DIR)
    try:
        df["split"]
    except KeyError:
        df["split"] = 0

    st.write(df.head())
    st.write(df.columns)

    if FILTER_SHORT:
        df = df.groupby(BY).filter(lambda x: len(x) > 20)

    samples = lib.utils.sample_groups(df, size=16, by=BY)
    fig, ax = plt.subplots(nrows=4, ncols=4)
    ax = ax.ravel()
    for n, (_, g) in enumerate(samples.groupby(BY)):
        plot_c0_c1(ax=ax[n], int_c0=g["int_c0"], int_c1=g["int_c1"])
        n_peaks = fc.number_peaks(g["int_c1"].values, n=5)
        ax[n].set_title(
            "rel = {:.1f}\npeaks = {}".format(
                g["int_c1"].max() / g["int_c1"].min(), n_peaks
            )
        )
    plt.tight_layout()
    st.write(fig)

    X_resampled = _process(df=df, path=INPUT_DIR, by=BY)

    st.subheader("Examples after resampling")
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.ravel()
    for n in range(len(ax)):
        xi = X_resampled[n]
        ax[n].plot(xi)
    plt.tight_layout()
    st.write(fig)
