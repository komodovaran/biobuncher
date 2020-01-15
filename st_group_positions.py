import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tiffile
import streamlit as st
from sklearn.neighbors import NearestNeighbors
import lib.utils

sns.set_style("dark")


def _load_cluster_sample_idx(path):
    """
    Loads the sample indices for each cluster and returns a dict.
    Must be paired with the correct pandas dataset.

    Args:
        path (str):
    """
    return pd.DataFrame(pd.read_hdf(path))


@st.cache
def _load_data(dataset_path):
    """
    Load same df as used for model fitting.

    Args:
        dataset_path (str):
    """
    return pd.DataFrame(pd.read_hdf(os.path.join(dataset_path)))


@st.cache
def _load_tiff_video(video_path):
    """
    Loads tiff file and returns video and shape.

    Args:
        video_path (str):
    """
    try:
        video = tiffile.imread(video_path)
        video = np.mean(video[0:5, ...], axis=0)
        video = video[..., 0]
    except FileNotFoundError:
        return None
    return video


@st.cache
def _find_groups(df, indices):
    """
    Finds the specific groups, based on provided indices

    Args:
        df (pd.DataFrame)
        indices (np.ndarray):
    """
    grouped_df = df.groupby(["file", "particle"])
    keys = list(grouped_df.groups.keys())
    groups = []
    for i in indices:
        grp = grouped_df.get_group(keys[i])
        groups.append(grp)
    return pd.concat(groups)


@st.cache
def _nearest_neighbour_dist(df):
    """
    Calculates nearest neighbour distribution for a given set of particles.
    First group by file to calculate distances for each video individually.
    Then pool all distances together (assumed to be distributed roughly the
    same for each video)

    Args:
        df (pd.DataFrame):
    """
    agg = df.groupby(["particle"]).apply(np.mean)
    xy = np.column_stack((agg["x"], agg["y"]))
    try:
        nbrs = NearestNeighbors(n_neighbors=2).fit(xy)
        distances, _ = nbrs.kneighbors(xy)
        distances = distances[:, 1]
    except ValueError:
        distances = np.array([])
    return distances


def _plot_trace_preview(df, nrows, ncols):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    ax = ax.ravel()
    for i, (_, grp) in enumerate(df.groupby(["file", "particle"])):
        if i == nrows * ncols:
            break
        ax[i].plot(grp["int_c0"], color="black")
        ax[i].plot(grp["int_c1"], color="red")
        ax[i].set_xticks(())
        ax[i].set_yticks(())
    st.write(fig)


def _plot_random_video_positions(df, video_dir, video_files, nrows, ncols):
    """
    Selects some random videos and plots the recorded coordinates

    Args:
        df (pd.DataFrame):
        video_dir (str):
        video_files (List[str]):
        nrows (int):
        ncols (int):
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    ax = ax.ravel()

    file_not_found_placeholder = st.empty()

    for i in range(len(video_files)):
        # Sub df
        sub = df[df["file"] == video_files[i]]

        # Find corresponding video
        video_file = "".join(
            lib.utils.split_and_keep(sub["file"].values[0], "/")[:-3]
        )
        video_file_path = os.path.join(video_dir, video_file, "RGB.tif")
        video = _load_tiff_video(video_file_path)

        # Select first row of every group
        agg = sub.groupby("particle").apply(np.mean)

        # Coordinates and nn dists
        xy = np.column_stack((agg["x"], agg["y"]))

        # ax[i].set_xticks(())
        # ax[i].set_yticks(())
        if video is None:
            ax[i].text(
                x=0.3, y=0.5, s="File not found!".format(video_file_path)
            )
            file_not_found_placeholder.subheader(
                "**File not found:** {}".format(video_file_path)
            )
            continue

        ax[i].imshow(video, cmap="Greys")
        ax[i].scatter(xy[:, 0], xy[:, 1], alpha=0.3, color="red")
        try:
            sns.kdeplot(
                xy[:, 0],
                xy[:, 1],
                shade=False,
                ax=ax[i],
                alpha=0.4,
                cmap="Reds",
            )
        except ValueError:
            print("Not enough samples to generate KDE. Skipped")

        ax[i].set_xlim(0, video.shape[1])
        ax[i].set_ylim(video.shape[0], 0)

    plt.tight_layout()
    st.write(fig)


def _plot_nearest_neighbour_dist(nn_dists):
    """
    Plots a histogram of the nearest neighbour distribution

    Args:
        nn_dists (np.ndarray):
    """
    bins = np.linspace(0, 150, 30)

    fig, ax = plt.subplots()
    ax.hist(nn_dists, bins = bins, color = "darkgrey", label = "N = {}".format(len(nn_dists)))
    ax.set_xlim(0, 150)
    ax.legend(loc = "upper right")

    st.write(fig)


def main():
    # Select indices
    indices_dir = "results/cluster_indices/"
    dataset_dir = "results/intensities/"
    top_video_dir = "/media/linux-data/Data/"

    st_indices_path = st.selectbox(
        options=sorted(glob(os.path.join(indices_dir, "*.h5"))),
        index=0,
        label="Select clustered indices",
        format_func=os.path.basename,
    )

    # Select dfc that is required to load indices from
    model_name = os.path.basename(st_indices_path.split("__")[0])

    # Remove the idx and h5 extension of the cluster index name to retrieve
    # df name
    df_name = st_indices_path.split("data=")[1].split("__")[0].split(".npz")[0]
    df_path = os.path.join(dataset_dir, "{}.h5".format(df_name))

    st.write("**Model**:")
    st.code(model_name)
    st.write("**Dataset**:")
    st.code(df_path)

    # Select a specific cluster
    cluster_sample_idx = _load_cluster_sample_idx(st_indices_path)
    cluster_label_set = list(cluster_sample_idx["cluster"].sort_values().unique())
    multi_index_data_names = list(cluster_sample_idx["file"].unique())

    if len(set(multi_index_data_names)) > 1:
        st.write("**Multi-index file found:** Dataset is composed of the following:")
        for name in multi_index_data_names:
            st.code(name)

    st_selected_cidx = st.selectbox(
        options=cluster_label_set, label="Select cluster", index = 0
    )
    st_selected_data_name = st.selectbox(
        options = multi_index_data_names, index = 0, label = "Select dataset")

    selected_cidx = cluster_sample_idx[(cluster_sample_idx["cluster"] == st_selected_cidx) & (cluster_sample_idx["file"] == st_selected_data_name)]
    selected_cidx = selected_cidx["idx"].values

    st.write(selected_cidx)
    st.write(max(selected_cidx))

    df = _load_data(dataset_path = os.path.join(dataset_dir, st_selected_data_name))
    dfc = _find_groups(df=df, indices=selected_cidx)
    _plot_trace_preview(dfc, nrows=5, ncols=5)

    select_mode = st.radio(
        label="Select videos...", options=["randomly", "from list"], index=0
    )

    if select_mode == "from list":
        files = st.multiselect(
            options=dfc["file"].unique(), label="Select 4 videos to plot"
        )
    else:
        video_names = dfc["file"].unique()
        files = np.random.choice(video_names, size=4)

    if len(files) == 4:
        _plot_random_video_positions(
            dfc, video_dir=top_video_dir, video_files=files, nrows=2, ncols=2
        )

    nn_dists = lib.utils.groupby_parallel_apply(
        grouped_df=dfc.groupby("file"), func=_nearest_neighbour_dist, concat = False
    )
    nn_dists = np.concatenate(nn_dists)

    _plot_nearest_neighbour_dist(nn_dists)


if __name__ == "__main__":
    main()
