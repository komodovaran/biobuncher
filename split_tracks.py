import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import sklearn.preprocessing
import streamlit as st
import lib.utils
import lib.plotting

@st.cache
def _get_data(path):
    return pd.DataFrame(pd.read_hdf(path))


def _split_ts(group):
    """Splits timeseries into new ones by detected peak locations"""
    if len(group) < WINDOW_LENGTH * 2:
        group
    else:
        s = group["int_c1"]
        s = s.clip(0)
        s = sklearn.preprocessing.minmax_scale(s)
        _, properties = scipy.signal.find_peaks(
            s, prominence=PROMINENCE, width=(5, 1000)
        )
        peaks = properties["right_bases"]

        if len(peaks) > 0:
            init = 0
            splits = []
            for i, p in enumerate(peaks):
                group["split"] = i
                splits.append(group[init:p])
                init = p
            group = pd.concat(splits)
        return group


if __name__ == "__main__":
    PATH = "results/intensities/tracks-cme.h5"

    df = _get_data(PATH)

    WINDOW_LENGTH = 11
    POLYORDER = 1
    PROMINENCE = (0.02, 1)  # 0.05, 1

    file = df["file"].unique()[0]

    fig, axes = plt.subplots(nrows=5, ncols=2)
    st.subheader("Examples of slicing using clathrin AND auxilin")
    for n in range(axes.shape[0]):
        sub = df[(df["file"] == file) & (df["particle"] == n)]
        lib.plotting.plot_c0_c1(
            sub["int_c0"], sub["int_c1"], ax=axes[n, 0], alpha=0.5
        )

        s = -(sub["int_c0"] * (sub["int_c1"] + sub["int_c0"]))
        s = scipy.signal.savgol_filter(
            s, window_length=WINDOW_LENGTH, polyorder=POLYORDER
        )
        s = sklearn.preprocessing.minmax_scale(s)

        axes[n, 1].plot(s)

        peaks, properties = scipy.signal.find_peaks(
            s, prominence=PROMINENCE, width=(5, 1000)
        )
        for p in peaks:
            axes[n, 0].axvline(p, color="black")
            axes[n, 1].axvline(p, color="red")
    plt.tight_layout()
    st.write(fig)

    fig, axes = plt.subplots(nrows=5, ncols=2)
    st.subheader("Examples of slicing using peaks and ONLY auxilin")
    for n in range(axes.shape[0]):
        sub = df[(df["file"] == file) & (df["particle"] == n)]
        lib.plotting.plot_c0_c1(
            sub["int_c0"], sub["int_c1"], ax=axes[n, 0], alpha=0.5
        )

        s = sub["int_c1"]
        s = s.clip(0)
        s = sklearn.preprocessing.minmax_scale(s)

        axes[n, 1].plot(s)

        _, properties = scipy.signal.find_peaks(s, prominence=(0.5, 1))

        peaks = properties.get("right_bases")

        for i, p in enumerate(peaks):
            axes[n, 0].axvline(p, color="black")
            axes[n, 1].axvline(p, color="red")

    plt.tight_layout()
    st.write(fig)

    split_df = lib.utils.groupby_parallel_apply(
        df.groupby(["file", "particle"]), _split_ts
    )
    split_df.to_hdf(PATH[:-3] + "_split-{}.h5".format("c1"), key = "df")

    st.write(split_df)

    fig, axes = plt.subplots(nrows=3)
    st.subheader(
        "Examples of slicing using clathrin AND auxilin after splits\n(splits shown in black)"
    )
    for n in range(axes.shape[0]):
        sub = split_df[(split_df["file"] == file) & (split_df["particle"] == n)]
        lib.plotting.plot_c0_c1(
            sub["int_c0"], sub["int_c1"], ax=axes[n], alpha=0.5
        )
        axes[n].twinx().plot(sub["split"], color="black")

    plt.tight_layout()
    st.write(fig)
