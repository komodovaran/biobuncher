import multiprocessing as mp
import os.path
from glob import glob
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parmap
import scipy.signal
import scipy.stats
import seaborn as sns
import skimage.color
import skimage.exposure
import skimage.io
import sklearn.preprocessing
import streamlit as st
import trackpy as tp
from configobj import ConfigObj

import lib.math
import lib.plotting
import lib.utils
from lib.utils import timeit

os.environ["NUMEXPR_MAX_THREADS"] = "72"
sns.set(context="notebook", style="darkgrid", palette="muted")
tp.quiet()


@st.cache
def _get_data(track_path, intensities_path):
    tracks = pd.DataFrame(pd.read_hdf((track_path)))
    roi_intensities_df = pd.DataFrame(pd.read_hdf(intensities_path))
    return tracks, roi_intensities_df


def _tiffpath(path):
    """
    Converts path to dynamic tiff path
    """
    return Path(path).parent.parent.joinpath("{}/*.tif").as_posix()


def remove_parent_dir(path, n):
    """
    Removes n directories from the left side of the path
    """
    return Path(*Path(path).parts[n + 1 :]).as_posix()

@timeit
def _get_videos(*pathlists):
    """Loads all videos in parallel and returns in same order and number of lists equal to input lists"""
    div = len(pathlists)
    paths = lib.utils.flatten_list(zip(*pathlists))
    with mp.pool.ThreadPool(mp.cpu_count()) as p:
        videos = p.map(skimage.io.imread, paths)
    rezipped = [videos[i::div] for i in range(div)]
    if len(rezipped) == 1:
        rezipped = rezipped[0]
    return rezipped


def _preprocess_videos(*videolists):
    """Subtracts bg from a list of videos"""
    div = len(videolists)
    videos = lib.utils.flatten_list(zip(*videolists))
    videos = [
        lib.math.fft_bg_video(
            v, K=int(CFG["FFT_K"]), percentile=int(CFG["FFT_P"])
        )
        for v in videos
    ]
    rezipped = [videos[i::div] for i in range(div)]
    return rezipped


def _get_features(video):
    """
    Set tracking parameters through tests
    """
    # keep it single threaded, because it's parallel per-video
    features = tp.batch(
        frames=video,
        diameter=int(CFG["TRACK_DIAMETER"]),
        minmass=np.mean(video) * float(CFG["TRACK_MINMASS_MULT"]),
        engine="numba",
        processes=os.cpu_count(),
    )
    return features


def _link_df(args):
    features, path = args
    save_name = remove_parent_dir(path, 1)
    tracks = tp.link_df(
        features,
        search_range=float(CFG["TRACK_RANGE"]),
        memory=int(CFG["TRACK_MEMORY"]),
    )
    tracks = tp.filter_stubs(tracks, threshold=int(CFG["LENGTH_THRESHOLD"]))
    # Add split for compatibility with CME trackswhich need to be split currently
    tracks["split"] = 0
    tracks["file"] = save_name
    return tracks


def _track(videos, paths):
    """
    Do the particle tracking
    """
    features = [_get_features(v) for v in videos]
    tracks = pd.concat(parmap.map(_link_df, zip(features, paths)), sort=False)
    tracks.to_hdf(RESULTS_PATH, key="df")
    return tracks


def _median_filter_sort(group):
    """
    Simple median filter for channel 1 signal to test if any big spikes
    """
    s = group["int_c1"]
    s = sklearn.preprocessing.minmax_scale(s)
    med = np.median(s)
    std = np.std(s)
    if np.any(s) > med + 3.5 * std:
        return group
    else:
        return None


def _n_peaks(group):
    """
    Count number of peaks in group
    """
    s = group["int_c1"]
    s = s.clip(0)
    s = sklearn.preprocessing.maxabs_scale(s)
    s = scipy.signal.medfilt(s, kernel_size=3)
    peaks, *_ = scipy.signal.find_peaks(s, prominence=(0.3, 1))
    return peaks, len(peaks)


def _SET_CONFIG():
    """
    Writes configs to a file, so they can be reloaded on next streamlit
    instance. They must be loaded as floats!
    """
    CFG["TRACK_MINMASS_MULT"] = TRACK_MINMASS_MULT
    CFG["TRACK_MEMORY"] = TRACK_MEMORY
    CFG["TRACK_DIAMETER"] = TRACK_DIAMETER
    CFG["TRACK_MEMORY"] = TRACK_MEMORY
    CFG["TRACK_RANGE"] = TRACK_RANGE
    CFG["LENGTH_THRESHOLD"] = LENGTH_THRESHOLD
    CFG["ROI_INNER"] = ROI_INNER
    CFG["ROI_OUTER"] = ROI_OUTER
    CFG["ROI_GAP"] = ROI_GAP
    CFG["FFT_K"] = FFT_K
    CFG["FFT_P"] = FFT_P
    CFG.write()


@st.cache
def _get_median_length(df):
    return lib.utils.groupby_parallel_apply(
        df.groupby(["file", "particle"]), _median_filter_sort
    )

@st.cache
def _get_n_peaks(df):
    return lib.utils.groupby_parallel_apply(
        df.groupby(["file", "particle"]), _n_peaks, concat=False
    )


if __name__ == "__main__":
    TIFFPATH = "data/kangmin_data/**/**/*.tif"
    RESULTS_PATH = "results/intensities/tracks-tpy.h5"
    CFG = ConfigObj("config/get_tracks.cfg")

    start = time()

    # Setup for the app
    st.sidebar.subheader(
        "Parameters (note: they don't do anything unless you press re-run)"
    )

    TRACK_MINMASS_MULT = st.sidebar.text_input(
        value=float(CFG["TRACK_MINMASS_MULT"]),
        label="Track Minimum Mass Multiplier",
    )

    TRACK_DIAMETER = st.sidebar.text_input(
        value=int(CFG["TRACK_DIAMETER"]),
        label="Track Spot Diameter (must be odd)",
    )
    TRACK_MEMORY = st.sidebar.text_input(
        value=int(CFG["TRACK_MEMORY"]), label="Track Memory"
    )
    TRACK_RANGE = st.sidebar.text_input(
        value=float(CFG["TRACK_RANGE"]), label="Track Search Range"
    )
    LENGTH_THRESHOLD = st.sidebar.text_input(
        value=int(CFG["LENGTH_THRESHOLD"]), label="Track Min Length"
    )
    ROI_INNER = st.sidebar.text_input(
        value=int(CFG["ROI_INNER"]), label="ROI Inner"
    )
    ROI_OUTER = st.sidebar.text_input(
        value=int(CFG["ROI_OUTER"]), label="ROI Outer"
    )
    ROI_GAP = st.sidebar.text_input(value=int(CFG["ROI_GAP"]), label="ROI Gap")
    FFT_K = st.sidebar.slider(
        min_value=1, max_value=30, value=int(CFG["FFT_K"]), label="FFT wiggle"
    )
    FFT_P = st.sidebar.slider(
        min_value=1, max_value=100, value=int(CFG["FFT_P"]), label="FFT noise"
    )

    RERUN = st.sidebar.button(
        label="Re-run tracking with updated parameters (takes a while)"
    )

    INTENSITY_PATH = RESULTS_PATH[:-3] + "_roi-int.h5"

    if RERUN or not st._is_running_with_streamlit:
        try:
            print(
                "Files deleted! Rerunning...\n"
                "Use streamlit next time if you don't want to re-run"
            )
            os.remove(RESULTS_PATH)
            os.remove(INTENSITY_PATH)
        except FileNotFoundError:
            pass
        st.caching.clear_cache()

    PROGRESS = 0
    PROGRESS_BAR = st.sidebar.progress(PROGRESS)

    try:
        tracks, roi_intensities_df = _get_data(RESULTS_PATH, INTENSITY_PATH)
        PROGRESS_BAR.progress(100)
    except FileNotFoundError:
        print("Loading and preprocessing videos...")
        paths_c0, paths_c1 = [
            _tiffpath(TIFFPATH).format(s) for s in ("TagRFP", "EGFP")
        ]
        paths_c0, paths_c1 = [
            sorted(glob(paths, recursive=True))
            for paths in (paths_c0, paths_c1)
        ]
        if len(paths_c0) != len(paths_c1):
            raise ValueError("Not enough videos for master/slave tracking")

        videos_c0, videos_c1 = _get_videos(paths_c0, paths_c1)
        videos_c0, videos_c1 = _preprocess_videos(videos_c0, videos_c1)
        mvideos = np.array(
            [
                np.stack((c0, c1), axis=-1)
                for (c0, c1) in zip(videos_c0, videos_c1)
            ]
        )
        indices = np.indices(mvideos[0, 0, ..., 0].shape)

        PROGRESS += 20
        PROGRESS_BAR.progress(PROGRESS)

        print("Tracking...")
        tracks = _track(videos=videos_c0, paths=paths_c0)
        PROGRESS += 20
        PROGRESS_BAR.progress(PROGRESS)

        print("Extracting intensities...")
        track_files = tracks["file"].unique()

        channels = mvideos.shape[-1]
        # Initialize columns
        keys = [("int_c{}".format(c)) for c in range(channels)]
        tracks = lib.utils.initialize_df_columns(df=tracks, new_columns=keys)

        roi_intensities = []
        for n, track_file_n in enumerate(track_files):
            tracks_n = tracks[tracks["file"] == track_file_n]

            def _get_intensities_applyf(group):
                """Find ROI intensities in parallel"""
                roi = np.zeros(shape=(len(group), channels))
                lib.math.calc_steplength(group, "x", "y")
                for i, row in enumerate(group.itertuples()):
                    y, x = row.y, row.x
                    frame = mvideos[n, row.frame, ...]
                    masks = lib.math.circle_mask(
                        yx=(y, x),
                        indices=indices,
                        gap_space=int(CFG["ROI_GAP"]),
                        inner_area=int(CFG["ROI_INNER"]),
                        outer_area=int(CFG["ROI_OUTER"]),
                    )
                    for c in range(channels):
                        roi[i, c] = lib.math.frame_roi_intensity(
                            frame[..., c], *masks
                        )
                group[keys] = roi
                return group

            roi_ints_n = lib.utils.groupby_parallel_apply(
                grouped_df=tracks_n.groupby("particle"),
                func=_get_intensities_applyf,
            )
            roi_intensities.append(roi_ints_n)
            PROGRESS += n * 5
            PROGRESS_BAR.progress(min(100, PROGRESS))

        roi_intensities_df = pd.concat(roi_intensities, sort=False)
        roi_intensities_df.to_hdf(INTENSITY_PATH, key="df")
        PROGRESS_BAR.progress(100)
        print("Done!")

    total_groups = roi_intensities_df.groupby(["file", "particle"]).ngroups
    has_aux = _get_median_length(roi_intensities_df)
    aux_track_lengths = has_aux.groupby(["file", "particle"]).apply(len)

    peaks, n_peaks = zip(*_get_n_peaks(has_aux))

    st.subheader("Total number of tracks")
    st.write(total_groups)

    st.subheader("May have auxilin")
    st.write(len(aux_track_lengths))

    nrows, ncols = 6, 5
    samples = []
    n = 0
    for _, group in has_aux.groupby(["file", "particle"]):
        if n == nrows * ncols:
            break
        samples.append(group)
        n += 1
    samples = pd.concat(samples)

    st.subheader(
        "Example tracks that *may* have auxilin (3.5 std from the median)"
    )
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    ax = ax.ravel()
    for n, (_, group) in enumerate(samples.groupby(["file", "particle"])):
        lib.plotting.plot_c0_c1(
            ax=ax[n], int_c0=group["int_c0"], int_c1=group["int_c1"]
        )
        ax[n].axhline(np.median(group["int_c1"]), ls="--", color="black")
        ax[n].set_title("length: {}".format(aux_track_lengths[n]))
        for p in peaks[n]:
            ax[n].axvline(p, color="black", alpha=0.5)
        ax[n].set_xticks(())
    plt.tight_layout()
    st.write(fig)

    st.subheader("Stats for tracks that may have auxilin")
    fig, ax = plt.subplots(ncols=2, figsize=(7, 2))
    bins = np.arange(0, 100, 5)
    ax[0].hist(aux_track_lengths, bins=bins, color="orange")
    ax[0].set_title("Length distribution")

    ax[1].hist(n_peaks, bins=np.arange(0, 10, 1) - 0.5)
    ax[1].set_xlim(-0.5, 4.5)
    ax[1].set_xticks([0, 1, 2, 3, 4])
    ax[1].set_title("Auxilin peaks per trace")

    plt.tight_layout()
    st.write(fig)

    _SET_CONFIG()
    end = time()

    print("Time elapsed: {:.1f} s".format(end - start))
