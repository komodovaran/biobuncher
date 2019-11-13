from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage.exposure
import skimage.io
import streamlit as st
import trackpy as tp
from configobj import ConfigObj

import lib.math

sns.set(context="notebook", style="darkgrid", palette="muted")
tp.quiet()


def _get_video(video_path):
    """Load video"""
    return skimage.io.imread(video_path)


np.random.seed(1)


@st.cache
def _get_tracks(video_frame):
    """Track single frame for preciew"""
    frame = (np.expand_dims(video_frame, axis=0),)
    tracks = tp.batch(
        frames=np.expand_dims(video_frame, axis=0),
        diameter=TRACK_DIAMETER,
        minmass=np.mean(video_frame) * TRACK_MINMASS_MULT,
        engine="numba",
        processes=8,
    )
    return tracks, np.squeeze(frame)


def _get_cmaps(color = "red"):
    """
    Colormaps for imshow
    """
    if color == "red":
        cm = plt.cm.Reds
    elif color == "green":
        cm = plt.cm.Greens
    else:
        raise NotImplementedError("no more colors")

    mask_cmap = cm
    mask_cmap.set_under("k", alpha=0)

    mask_cmap_bg = cm
    mask_cmap_bg.set_under("k", alpha=0)

    return mask_cmap, mask_cmap_bg


def _tiffpath(path):
    """
    Converts path to dynamic tiff path
    """
    return Path(path).parent.parent.joinpath("{}/*.tif").as_posix()


def _single_frame(video, frame):
    return np.expand_dims(video[frame, ...], axis=0)


def _enhance_contrast(image, contrast=2):
    return image.clip(0, image.max() // contrast)


def _SET_CONFIG():
    """
    Writes configs to a file, so they can be reloaded on next streamlit
    instance. They must be loaded as floats!
    """
    CFG["TRACK_MINMASS_MULT"] = TRACK_MINMASS_MULT
    CFG["TRACK_DIAMETER"] = TRACK_DIAMETER
    CFG["ROI_INNER"] = ROI_INNER
    CFG["ROI_OUTER"] = ROI_OUTER
    CFG["ROI_GAP"] = ROI_GAP
    CFG["FFT_K"] = FFT_K
    CFG["FFT_P"] = FFT_P
    CFG.write()


if __name__ == "__main__":
    TIFFPATH = "data/kangmin_data/**/**/*.tif"
    RESULTS_PATH = "results/intensities/tracks-tpy.h5"
    CFG = ConfigObj("config/get_tracks.cfg")

    paths_egfp, paths_tagrfp = [
        _tiffpath(TIFFPATH).format(s) for s in ("EGFP", "TagRFP")
    ]
    tiffs_egfp, tiffs_tagrfp = [
        sorted(glob(paths, recursive=True))
        for paths in (paths_egfp, paths_tagrfp)
    ]
    SWITCH_C = st.sidebar.checkbox(label = "Switch tracking channel", value = False)

    VIDEO_IDX = st.sidebar.slider(
        min_value=0,
        max_value=len(tiffs_egfp) - 1,
        value=0,
        label="Video Index Preview",
    )

    if SWITCH_C:
        video_c0 = _get_video(video_path = tiffs_egfp[VIDEO_IDX])
        video_c1 = _get_video(video_path = tiffs_tagrfp[VIDEO_IDX])
    else:
        video_c0 = _get_video(video_path=tiffs_tagrfp[VIDEO_IDX])
        video_c1 = _get_video(video_path=tiffs_egfp[VIDEO_IDX])

    FRAME = st.sidebar.slider(
        min_value=0, max_value=video_c0.shape[0], value=0, label="Frame Preview"
    )

    TRACK_MINMASS_MULT = st.sidebar.number_input(
        value=float(CFG["TRACK_MINMASS_MULT"]),
        label="track minmass multiplier",
    )

    TRACK_DIAMETER = st.sidebar.number_input(
        value=int(CFG["TRACK_DIAMETER"]),
        label="Track Spot Diameter (must be odd)",
    )
    ROI_INNER = st.sidebar.number_input(
        value=int(CFG["ROI_INNER"]), label= "ROI Inner"
    )
    ROI_OUTER = st.sidebar.number_input(
        value=int(CFG["ROI_OUTER"]), label= "ROI Outer"
    )
    ROI_GAP = st.sidebar.number_input(
        value=int(CFG["ROI_GAP"]), label= "ROI Gap"
    )
    ROI_INNER_ALPHA = st.sidebar.slider(
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        label="ROI Inner Alpha",
    )
    ROI_OUTER_ALPHA = st.sidebar.slider(
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        label="ROI Outer Alpha",
    )

    FFT_K = st.sidebar.slider(
        min_value=1, max_value=30, value=int(CFG["FFT_K"]), label= "FFT wiggle"
    )
    FFT_P = st.sidebar.slider(
        min_value=1, max_value=100, value=int(CFG["FFT_P"]), label= "FFT noise"
    )

    _SET_CONFIG()

    props = {}
    for n, video in enumerate((video_c0, video_c1)):
        props["min_c{}".format(n)] = np.min(video, axis=(0, 1, 2))
        props["mean_c{}".format(n)] = np.mean(video, axis=(0, 1, 2))
        props["max_c{}".format(n)] = np.max(video, axis=(0, 1, 2))

    tracks, frame_c0 = _get_tracks(video_c0[FRAME, ...])
    frame_c1 = video_c1[FRAME, ...]

    indices = np.indices(frame_c0.shape)
    ctr_masks = np.zeros(frame_c0.shape)
    bg_masks = np.zeros(frame_c0.shape)

    for row in tracks.itertuples():
        x, y = row.x, row.y
        ctr_mask, bg_mask = lib.math.circle_mask(
            yx=(y, x),
            inner_area=ROI_INNER,
            outer_area=ROI_OUTER,
            gap_space=ROI_GAP,
            indices=indices,
        )
        ctr_masks += ctr_mask
        bg_masks += bg_mask

    ctr_cmap, bg_cmap = _get_cmaps()
    random_rows = np.random.randint(0, frame_c0.shape[0], 9)
    fig, ax = plt.subplots(nrows=2)
    st.subheader("Without background correction")
    for n, frame in enumerate((frame_c0, frame_c1)):
        ax[n].imshow(_enhance_contrast(frame, 2), cmap="Greys")
        ax[n].imshow(
            ctr_masks, cmap=ctr_cmap, clim=(0.5, 1), alpha=ROI_INNER_ALPHA
        )
        ax[n].imshow(
            bg_masks, cmap=bg_cmap, clim=(0.5, 1), alpha=ROI_OUTER_ALPHA
        )
        if n == 0:
            for idx in random_rows:
                ax[n].axhline(idx, color="blue", alpha=0.5, ls="--")
    plt.tight_layout(h_pad=0, w_pad=0)
    st.write(fig)

    frame_c0_bg, frame_c1_bg = [
        lib.math.fft_bg_2d(f, K=FFT_K, percentile=FFT_P)
        for f in (frame_c0, frame_c1)
    ]

    st.subheader("With FFT background correction")
    fig, ax = plt.subplots(nrows=2)
    for n, (frame, bg) in enumerate(
        zip((frame_c0, frame_c1), (frame_c0_bg, frame_c1_bg))
    ):
        corrected = frame - bg

        ax[n].imshow(_enhance_contrast(corrected, 4), cmap="Greys")
        ax[n].imshow(
            ctr_masks, cmap=ctr_cmap, clim=(0.5, 1), alpha=ROI_INNER_ALPHA
        )
        ax[n].imshow(
            bg_masks, cmap=bg_cmap, clim=(0.5, 1), alpha=ROI_OUTER_ALPHA
        )

    plt.tight_layout()
    st.write(fig)

    for i, (frame, bg) in enumerate(
        ((frame_c0, frame_c0_bg), (frame_c1, frame_c1_bg))
    ):
        st.subheader("FFT correction for random rows in C{}".format(i))
        fig, ax = plt.subplots(nrows=3, ncols=3)
        ax = ax.ravel()
        for j, row_idx in enumerate(random_rows):
            ax[j].plot(frame[row_idx], color = "lightblue")
            ax[j].plot(bg[row_idx], color = "black", alpha = 0.5)
            ax[j].plot(frame[row_idx] - bg[row_idx], color = "orange")
            ax[j].set_xticks(())
            ax[j].set_yticks(())
        plt.tight_layout()
        st.write(fig)

    intensities = []
    for frame in (frame_c0, frame_c0 - frame_c0_bg):
        r = np.zeros((len(tracks), 1))
        for i, row in enumerate(tracks.itertuples()):
            x, y = row.x, row.y
            ctr_mask, bg_mask = lib.math.circle_mask(
                yx=(y, x),
                inner_area=ROI_INNER,
                outer_area=ROI_OUTER,
                gap_space=ROI_GAP,
                indices=indices,
            )
            r[i] = lib.math.frame_roi_intensity(frame, ctr_mask, bg_mask)
        intensities.append(r)

    st.subheader(
        "Spot intensity distribution before/after FFT background correction"
    )
    bins = np.arange(0, 2000, 100)

    fig, ax = plt.subplots(ncols=2)
    p = dict(bins=bins, alpha=0.5, density=False)

    ax[0].hist(intensities[0], color = "salmon", **p)
    ax[0].set_yticks(())

    ax[1].hist(intensities[1], color = "seagreen", **p)
    ax[1].set_yticks(())

    plt.tight_layout()
    st.write(fig)
