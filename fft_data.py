import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import lib.math
import lib.utils


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
    X = np.zeros((grouped_df.ngroups, max_len, 2))
    for n, (_, group) in enumerate(grouped_df):
        l = group.values.shape[0]
        X[n, 0:l, :] = group[["int_c0", "int_c1"]]
    # Don't run in parallel, too fast
    X_fft = np.array([lib.math.nd_fft_ts(xi, log_transform = True) for xi in X])
    if not st._is_running_with_streamlit:
        np.savez(path[:-3] + "_fft.npz", data=X_fft)
    return X, X_fft


if __name__ == "__main__":
    INPUT_DIR = "results/intensities/tracks-tpy_roi-int.h5"
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
        Xi = X[idx, 0:100]
        ax[n].plot(Xi)
    plt.tight_layout()
    st.write(fig)

    fig, ax = plt.subplots(nrows = 4, ncols = 4)
    ax = ax.ravel()
    for n, idx in enumerate(samples):
        ax[n].plot(np.log(1+X_fft[idx]))

    plt.tight_layout()
    st.write(fig)
    st.write(df.head())
