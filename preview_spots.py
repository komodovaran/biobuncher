from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure
import skimage.io
import streamlit as st
import trackpy as tp
from configobj import ConfigObj
import scipy.interpolate
import lib.config as c
import lib.math
from glob import glob
import seaborn as sns

sns.set(context = "notebook", style = "darkgrid", palette = "muted")
tp.quiet()
np.random.seed(1)


def _get_video(video_path):
    return skimage.io.imread(video_path)


@st.cache
def _get_tracks(video_frame):
    frame = (np.expand_dims(video_frame, axis = 0),)
    tracks = tp.batch(
        frames = np.expand_dims(video_frame, axis = 0),
        diameter = TRACK_DIAMETER,
        minmass = np.mean(video_frame) * TRACK_MINMASS_MULT,
        engine = "numba",
        processes = 8,
    )
    return tracks, np.squeeze(frame)


def _get_cmaps():
    """
    Colormaps for imshow
    """
    mask_cmap = plt.cm.Reds
    mask_cmap.set_under("k", alpha = 0)

    mask_cmap_bg = plt.cm.Reds
    mask_cmap_bg.set_under("k", alpha = 0)

    return mask_cmap, mask_cmap_bg


def _tiffpath(path):
    """
    Converts path to dynamic tiff path
    """
    return Path(path).parent.parent.joinpath("{}/*.tif").as_posix()


def _single_frame(video, frame):
    return np.expand_dims(video[frame, ...], axis = 0)


def _SET_CONFIG():
    """
    Writes configs to a file, so they can be reloaded on next streamlit
    instance. They must be loaded as floats!
    """
    CONFIG["TRACK_MINMASS_MULT"] = TRACK_MINMASS_MULT
    CONFIG["TRACK_DIAMETER"] = TRACK_DIAMETER
    CONFIG["ROI_INNER"] = ROI_INNER
    CONFIG["ROI_OUTER"] = ROI_OUTER
    CONFIG["ROI_GAP"] = ROI_GAP
    CONFIG.write()


if __name__ == "__main__":
    TIFFPATH = "data/kangmin_data/**/**/*.tif"
    RESULTS_PATH = "results/intensities/tracks-tpy.h5"
    CONFIG = ConfigObj("config/get_tracks.cfg")

    paths_egfp, paths_tagrfp = [
        _tiffpath(TIFFPATH).format(s) for s in ("EGFP", "TagRFP")
    ]
    tiffs_egfp, tiffs_tagrfp = [
        sorted(glob(paths, recursive = True))
        for paths in (paths_egfp, paths_tagrfp)
    ]

    VIDEO_IDX = st.sidebar.slider(min_value = 0, max_value = len(tiffs_egfp) - 1, value = 0,
                                  label = "Video Index Preview")

    video_c0 = _get_video(video_path = tiffs_tagrfp[0])
    video_c1 = _get_video(video_path = tiffs_egfp[0])

    FRAME = st.sidebar.slider(
        min_value = 0, max_value = video_c0.shape[0], value = 0, label = "Frame Preview"
    )

    TRACK_MINMASS_MULT = st.sidebar.number_input(
        value = float(CONFIG["TRACK_MINMASS_MULT"]),
        label = "track minmass multiplier",
    )

    TRACK_DIAMETER = st.sidebar.number_input(
        value = int(CONFIG["TRACK_DIAMETER"]),
        label = "Track Spot Diameter (must be odd)",
    )
    ROI_INNER = st.sidebar.number_input(
        value = int(CONFIG["ROI_INNER"]), label = "ROI Inner"
    )
    ROI_OUTER = st.sidebar.number_input(
        value = int(CONFIG["ROI_OUTER"]), label = "ROI Outer"
    )
    ROI_GAP = st.sidebar.number_input(
        value = int(CONFIG["ROI_GAP"]), label = "ROI Gap"
    )

    ROI_ALPHA = st.sidebar.slider(
        min_value = 0.0,
        max_value = 1.0,
        value = 0.2,
        step = 0.1,
        label = "ROI Display Alpha",
    )

    props = {}
    for n, video in enumerate((video_c0, video_c1)):
        props["min_c{}".format(n)] = np.min(video, axis = (0, 1, 2))
        props["mean_c{}".format(n)] = np.mean(video, axis = (0, 1, 2))
        props["max_c{}".format(n)] = np.max(video, axis = (0, 1, 2))

    tracks, frame_c0 = _get_tracks(video_c0[FRAME, ...])
    frame_c1 = video_c1[FRAME, ...]

    indices = np.indices(frame_c0.shape)
    ctr_masks = np.zeros(frame_c0.shape)
    bg_masks = np.zeros(frame_c0.shape)

    for i in tracks.itertuples():
        x, y = i.x, i.y
        ctr_mask, bg_mask = lib.math.circle_mask(
            yx = (y, x),
            inner_area = c.ROI_INNER,
            outer_area = c.ROI_OUTER,
            gap_space = c.ROI_GAP,
            indices = indices,
        )
        roi_intensity, bg_intensity = lib.math.frame_roi_intensity(
            frame_c0, roi_mask = ctr_mask, bg_mask = bg_mask
        )

        ctr_masks += ctr_mask
        bg_masks += bg_mask

    ctr_cmap, bg_cmap = _get_cmaps()
    random_rows = np.random.randint(0, frame_c0.shape[0], 4)
    fig, ax = plt.subplots(ncols = 2)
    for n, frame in enumerate((frame_c0, frame_c1)):
        ax[n].imshow(frame.clip(0, frame.mean()*2), cmap = "Greys")
        ax[n].imshow(ctr_masks, cmap = ctr_cmap, clim = (0.5, 1), alpha = ROI_ALPHA)
        ax[n].imshow(bg_masks, cmap = bg_cmap, clim = (0.5, 1), alpha = ROI_ALPHA)

        if n == 0:
            for idx in random_rows:
                ax[n].axhline(idx, color = "blue")

    plt.tight_layout()
    st.write(fig)

    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    ax = ax.ravel()
    frame_c0_bg = lib.math.smooth_spline_2d(image = frame_c0, kx = 3, ky = 3, s = 7e9)
    st.subheader("Spline smoothing for different rows")
    for n, row_idx in enumerate(random_rows):
        ax[n].plot(frame_c0[row_idx])
        ax[n].plot(frame_c0_bg[row_idx])
        ax[n].set_ylim(0, np.max(frame_c0))
        ax[n].set_xticks(())
        ax[n].set_yticks(())
    plt.tight_layout()
    st.write(fig)

    # fig, ax = plt.subplots(ncols = 2)
    # for n, frame in enumerate((frame_c0, frame_c1)):
    #     bg = lib.math.smooth_spline_2d(image = frame, kx = 3, ky = 3, by = "both", s = 3e8)
    #     ax[n].imshow(bg, cmap = "plasma_r")
    #     # ax.imshow(ctr_masks, cmap = ctr_cmap, clim = (0.5, 1), alpha = ROI_ALPHA)
    #     # ax.imshow(bg_masks, cmap = bg_cmap, clim = (0.5, 1), alpha = ROI_ALPHA)
    # plt.tight_layout()
    # st.write(fig)