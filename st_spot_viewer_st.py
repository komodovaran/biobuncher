import os.path
import os.path
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tiffile
import lib.plotting
import lib.math
from lib.utils import get_single_df_group, split_and_keep


def _get_cmaps(color="red"):
    """
    Colormaps for imshow
    """
    if color == "red":
        cm = plt.cm.Reds
    elif color == "green":
        cm = plt.cm.Greens
    elif color == "blue":
        cm = plt.cm.Blues
    elif color == "white":
        cm = plt.cm.Greys_r
    elif color == "black":
        cm = plt.cm.Greys
    else:
        raise NotImplementedError("no more colors")

    mask_cmap = cm
    mask_cmap.set_under("k", alpha=0)

    mask_cmap_bg = cm
    mask_cmap_bg.set_under("k", alpha=0)

    return mask_cmap, mask_cmap_bg


def _select_dataset(dataset_dir):
    """Select dataset"""
    datasets = sorted(glob(dataset_dir + "/*.h5"))
    dataset = st.sidebar.selectbox(
        options=datasets,
        index=len(datasets) - 1,
        label="Select dataframe",
        format_func=os.path.basename,
    )
    return dataset


@st.cache
def _load_data(dataset_dir, trace_id):
    """
    Load same df as used for model fitting.
    """
    df = pd.DataFrame(pd.read_hdf(os.path.join(dataset_dir)))
    grp = get_single_df_group(df, trace_id, by=["file", "particle"])
    return grp


@st.cache
def _load_tiff_video(path):
    """
    Loads tiff file and returns video and shape.
    """
    video = tiffile.imread(path)
    t, h, w, d = video.shape
    return video, (t, h, w, d)


def _plot_trace(time, values, colors, frame_number):
    fig, ax = plt.subplots()
    for i, val in enumerate(values):
        ax.plot(time, val, color=colors[i])
    ax.axvline(frame_number, color="black", ls="--")
    ax.set_xlim(time.min(), time.max())
    if len(time) < 15:
        ax.set_xticks(np.arange(min(time), max(time), 3))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Intensity")
    st.write(fig)


def _zoomed_pos(x, y, pad=30):
    """
    Returns zoomed view for plt.imshow with default origin. Set with xlim/ylim.
    """
    x_left = int(x - pad)
    x_right = int(x + pad)
    y_top = int(y + pad)
    y_bottom = int(y - pad)
    return x_left, x_right, y_top, y_bottom


def _arrow_params(x, y):
    """
    Parameters for plt.text. Use like plt.text(**params)
    """
    params = dict(
        x=x + 20,
        y=y - 20,
        s="     ",
        ha="center",
        va="center",
        rotation=45 * 5,
        size=4,
        bbox=dict(
            boxstyle="rarrow,pad=0.3", fc="white", ec="black", lw=1, alpha=1
        ),
    )
    return params


def _plot_roi(x, y, img_h, img_w, axes, color):
    """
    Creates a ROI ring for a single frame, given the coordinates and frame shape
    """
    bg_masks = np.zeros((img_h, img_w))
    indices = np.indices((img_h, img_w))
    _, bg_mask = lib.math.circle_mask(
        yx=(y, x), inner_area=5, outer_area=9, gap_space=5, indices=indices,
    )
    bg_masks += bg_mask
    _, bg_cmap = _get_cmaps(color=color)
    for ax in axes:
        ax.imshow(bg_masks, cmap=bg_cmap, clim=(0.5, 1), alpha=0.6)


def _plot_frame(video, x_zoom_pos, y_zoom_pos, frame_number):
    """
    Plots a single frame from a video, and zooms in at a given position
    """
    idx = frame_number - 1

    x0, x1, y0, y1 = _zoomed_pos(x=x_zoom_pos, y=y_zoom_pos, pad=30)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    axes = axes.ravel()

    for c in [0, 1]:
        # normal axes
        axes[c].set_title("Channel {}".format(c))
        axes[c].imshow(video[idx, ..., c], cmap="Greys")
        axes[c].text(**_arrow_params(x_zoom_pos, y_zoom_pos))
        axes[c].set_xticks(())
        axes[c].set_yticks(())

        # zoomed axes
        axes[c + 2].set_title("Channel {} zoom".format(c))
        axes[c + 2].imshow(video[idx, ..., 0], cmap="Greys")
        axes[c + 2].set_xlim(x0, x1)
        axes[c + 2].set_ylim(y0, y1)

    return fig, axes


def main():
    dataset_dir = "results/intensities"
    top_video_dir = "/media/linux-data/Data"

    trace_id = st.sidebar.number_input(
        min_value=0, max_value=int(1e6), value=0, label="Trace ID"
    )

    dataset = _select_dataset(dataset_dir)
    grp = _load_data(dataset, trace_id)

    if "x" not in grp.columns:
        quit("No coordinates found in dataset. Maybe it's not included?")

    int_c0 = grp["int_c0"].values
    int_c1 = grp["int_c1"].values
    x_coords = grp["x"].values
    y_coords = grp["y"].values
    real_time = grp["t"].values
    cme_file_path = grp["file"][0]

    x_mean = int(np.mean(x_coords))
    y_mean = int(np.mean(y_coords))

    video_dir = "".join(split_and_keep(cme_file_path, "/")[:-3])
    video_file_path = os.path.join(top_video_dir, video_dir, "RGB.tif")

    video, (tmax, h, w, d) = _load_tiff_video(video_file_path)

    # index_time = np.arange(0, len(real_time), 1)
    first_frame = int(real_time[0])
    last_frame = int(real_time[-1])

    frame_number = st.sidebar.number_input(
        min_value=1, max_value=int(tmax), label="Video frame", step=1
    )

    idx = int(frame_number - first_frame)

    if idx < 0:
        idx = 0
    elif idx >= len(real_time):
        idx = len(real_time) - 1
    else:
        pass

    st.write("**CME file:**", cme_file_path)
    st.write("**Video file:**", video_file_path)
    st.write(
        "**First frame**: {:d}".format(int(first_frame)),
        "**Last frame**: {:d}".format(int(last_frame)),
        "**Movie length**: {:d}".format(int(tmax)),
    )

    _plot_trace(
        time=real_time,
        values=[int_c0, int_c1],
        colors=["black", "red"],
        frame_number=frame_number,
    )

    fig, axes = _plot_frame(
        video=video,
        x_zoom_pos=x_mean,
        y_zoom_pos=y_mean,
        frame_number=frame_number,
    )
    x_ = int(x_coords[idx])
    y_ = int(y_coords[idx])
    if frame_number < first_frame or frame_number > last_frame:
        roi_c = "blue"
    else:
        roi_c = "red"
    _plot_roi(x=x_, y=y_, img_h=h, img_w=w, axes=axes, color=roi_c)
    st.write(fig)


if __name__ == "__main__":
    main()
