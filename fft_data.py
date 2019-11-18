import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import lib.math
import lib.utils
import lib.plotting

@st.cache
def _get_data(path):
    """
    Loads all traces and converts them to a padded tensor
    """
    df = pd.DataFrame(pd.read_hdf(path))
    return df

@st.cache
def _process(df, path, by):
    """
    Zero-pad everything to match max-length of video, then do fourier transform
    """
    grouped_df = df.groupby(by)
    max_len = 300 # video max length
    X_padded = np.zeros((grouped_df.ngroups, max_len, 2))
    X_resampled = X_padded.copy()
    for n, (_, group) in enumerate(grouped_df):
        l = group.values.shape[0]
        X_padded[n, 0:l, :] = group[["int_c0", "int_c1"]]
        X_resampled[n, ...] = lib.math.resample_timeseries(group[["int_c0", "int_c1"]].values, new_length = max_len)

    # padded
    np.savez(path[:-3] + "_pad.npz", data = X_padded)

    # resampled
    np.savez(path[:-3] + "_res.npz", data = X_resampled)

    # fft
    X_fft = np.array([lib.math.nd_fft_ts(xi, log_transform = True) for xi in X_resampled])
    X_fft = X_fft[:, :80, :]
    np.savez(path[:-3] + "_fft-{}.npz".format(X_fft.shape[1]), data=X_fft)
    return X_padded, X_fft


if __name__ == "__main__":
    INPUT_DIR = "results/intensities/tracks-cme_split-c1.h5"
    BY = ["file", "particle", "split"]
    FILTER_SHORT = 10

    df = _get_data(INPUT_DIR)

    if "split" not in df.columns:
        df["split"] = np.zeros(len(df))

    if FILTER_SHORT:
        df = df.groupby(BY).filter(lambda x: len(x) > 20)

    X, X_fft = _process(df=df, path=INPUT_DIR, by=BY)
    samples = np.random.randint(0, len(X), 16)

    st.subheader("Examples")
    fig, ax = plt.subplots(nrows = 4, ncols = 4)
    ax = ax.ravel()
    for n, idx in enumerate(samples):
        xi = X[idx, 0:100]
        xi0 = np.trim_zeros(xi[:, 0].ravel())
        xi1 = np.trim_zeros(xi[:, 1].ravel())

        ax[n].plot(xi0)
        ax[n].plot(xi1)
    plt.tight_layout()
    st.write(fig)

    fig, ax = plt.subplots(nrows = 4, ncols = 4)
    ax = ax.ravel()
    for n, idx in enumerate(samples):
        ax[n].plot(np.log(1+X_fft[idx]))

    plt.tight_layout()
    st.write(fig)
    st.write(df.head())