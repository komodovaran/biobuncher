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
import lib.globals
from lib.plotting import svg_write

sns.set_style("dark")


@st.cache
def _load_cluster_sample_idx(path):
    """
    Loads the sample indices for each cluster and returns a dict.
    Must be paired with the correct pandas dataset.

    Args:
        path (str):
    """
    return pd.DataFrame(pd.read_hdf(path))


@st.cache
def _load_df(path):
    """
    Load same df as used for model fitting.

    Args:
        path (str):
    """
    return pd.DataFrame(pd.read_hdf(os.path.join(path)))


@st.cache
def _load_npz(path):
    """
    Loads all traces

    Args:
        path (str)
    """
    return np.load(path, allow_pickle=True)["data"]


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


def _plot_df_trace_preview(df, nrows, ncols, colors):
    """
    Plots trace previews

    Args:
        df (pd.DataFrame):
        nrows (int):
        ncols (int):
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    ax = ax.ravel()
    for i, (_, grp) in enumerate(df.groupby(lib.globals.groupby)):
        if i == nrows * ncols:
            break

        columns = df.columns[[c.startswith("int_") for c in df.columns]]
        for j, c in enumerate(columns):
            ax[i].plot(grp[c], alpha=0.5, color=colors[j])
        ax[i].set_title("L = {}".format(len(grp)))
        ax[i].set_xticks(())
        ax[i].set_yticks(())
    plt.tight_layout()
    svg_write(fig)


def _plot_trace_preview(X, nrows, ncols, colors):
    """
    Plots trace previews

    Args:
        df (pd.DataFrame):
        nrows (int):
        ncols (int):
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    ax = ax.ravel()
    for i, ax in enumerate(ax):
        xi = X[i]
        for c in range(xi.shape[-1]):
            ax.plot(xi[:, c], color=colors[c])
        ax.set_xticks(())
        ax.set_yticks(())
    svg_write(fig)


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


def _plot_nearest_neighbour_dist(nn_dists, n_videos):
    """
    Plots a histogram of the nearest neighbour distribution

    Args:
        nn_dists (np.ndarray):
    """
    bins = np.linspace(0, 150, 30)

    fig, ax = plt.subplots()
    ax.hist(
        nn_dists,
        bins=bins,
        color="darkgrey",
        label="N = {}, ({} videos)".format(len(nn_dists), n_videos),
        density=True,
    )
    ax.set_xlim(0, 150)
    ax.legend(loc="upper right")
    ax.set_xlabel("Distance to nearest neighbour")
    ax.set_ylabel("Density")

    svg_write(fig)


def main():
    # Select indices
    st_indices_path = st.selectbox(
        options=sorted(glob(os.path.join(lib.globals.cluster_idx_dir, "*.h5"))),
        index=0,
        label="Select clustered indices",
        format_func=os.path.basename,
    )

    # Select dfc that is required to load indices from
    model_name = os.path.basename(st_indices_path.split("__")[0])

    # Remove the idx and h5 extension of the cluster index name to retrieve
    # df name
    data_name = (
        st_indices_path.split("data=")[1].split("___")[1].split("__")[1][:-4]
    )

    df_path = os.path.join(
        lib.globals.data_preprocessed_dir, "{}.h5".format(data_name)
    )
    npz_path = os.path.join(
        lib.globals.data_preprocessed_dir, "{}.npz".format(data_name)
    )
    npz_name = os.path.basename(npz_path)

    st.write("**Model**:")
    st.code(model_name)
    st.write("**Dataset**:")
    st.code(df_path)

    # df = _load_df(path=df_path)
    try:
        df = _load_df(df_path)
        source_names = df["source"].unique()
        if len(source_names) > 1:
            if len(source_names) > 1:
                st.write(
                    "**Multi-index file found:** Dataset is composed of the following:"
                )
                for name in source_names:
                    st.code(name)

        # Select dataset
        st_selected_data_name = st.multiselect(
            options=source_names, label="Select dataset"
        )
        if not st_selected_data_name:
            return
        df = df[df["source"].isin(st_selected_data_name)]
    except FileNotFoundError:
        df = None

    X = _load_npz(npz_path)

    # Select clusters
    idx_file = _load_cluster_sample_idx(st_indices_path)
    cluster_label_set = list(idx_file["cluster"].sort_values().unique())
    st_selected_cidx = st.multiselect(
        options=cluster_label_set, label="Select cluster"
    )
    if not st_selected_cidx:
        return

    # Get group IDs from dataset
    cidx = idx_file[idx_file["cluster"].isin(st_selected_cidx)]["id"].values

    Xc = X[cidx]

    # This df contains all the clustered samples
    _plot_trace_preview(Xc, colors=["red", "black", "blue"], nrows=5, ncols=5)

    if df is not None:
        dfc = df[df["id"].isin(cidx)]
        video_names = dfc["file"].unique()

        select_mode = st.radio(
            label="Select videos...", options=["randomly", "from list"], index=0
        )

        n_req_vids = 4
        if select_mode == "from list":
            files = st.multiselect(
                options=video_names,
                label="Select {} videos to plot".format(n_req_vids),
            )
        else:
            files = np.random.choice(video_names, size=n_req_vids)

        if len(files) == n_req_vids:
            _plot_random_video_positions(
                dfc,
                video_dir=lib.globals.top_video_dir,
                video_files=files,
                nrows=2,
                ncols=2,
            )

        video_grouping = dfc.groupby("file")

        nn_dists = lib.utils.groupby_parallel_apply(
            grouped_df=video_grouping,
            func=_nearest_neighbour_dist,
            concat=False,
        )
        nn_dists = np.concatenate(nn_dists)

        st.write(
            "Counts may be lower than number of samples,"
            "if video contained only 1 of a particular trace"
        )
        _plot_nearest_neighbour_dist(
            nn_dists=nn_dists, n_videos=len(video_names)
        )
    else:
        dfc = None

    if st.button(label="Save selection to file"):
        np.savez(
            (os.path.join(lib.globals.data_preprocessed_dir, "saved.npz")),
            data=Xc,
        )

        classifier_labels = np.zeros(len(X))
        classifier_labels[cidx] = 1
        np.savez(
            os.path.join(
                lib.globals.labels_dir,
                npz_name[:-4] + "__clust_{}.npz".format(st_selected_cidx),
            ),
            y=classifier_labels,
        )

        if dfc is not None:
            dfc.reset_index()
            dfc["id"] = dfc.groupby(lib.globals.groupby_src).ngroup()

            dfc.to_hdf(
                os.path.join(lib.globals.data_preprocessed_dir, "saved.h5"),
                key="df",
            )

        st.write("New files saved!")


if __name__ == "__main__":
    main()
