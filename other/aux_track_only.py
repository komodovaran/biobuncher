import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parmap
import streamlit as st
import trackpy as tp
from configobj import ConfigObj

import lib.math
import lib.plotting
import lib.utils
from other.get_track_positions import _get_videos, _tiffpath


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
def _get_data(intensities_path):
    roi_intensities_df = pd.DataFrame(pd.read_hdf(intensities_path))
    return roi_intensities_df


@st.cache
def _preprocess_videos(videolist, **fft_kwargs):
    """Subtracts bg from a list of videos"""
    videos = [
        lib.math.fft_bg_video(
            video=v, **fft_kwargs
        )
        for v in videolist
    ]
    return videos

@st.cache
def _get_detections(video, diameter, min_mass_mult):
    """
    Set tracking parameters through tests
    """
    detections_df = tp.batch(
        frames=video,
        diameter=diameter,
        minmass=np.mean(video) * min_mass_mult,
        engine="numba",
        processes=os.cpu_count(),
    )
    return detections_df

@st.cache
def _link_df(args):
    """
    Link df for every movie in parallel
    """
    detections_df, path = args
    save_name = lib.utils.remove_parent_dir(path, 1)
    tracks = tp.link_df(
        detections_df,
        search_range=float(CFG["TRACK_RANGE"]),
        memory=int(CFG["TRACK_MEMORY"]),
    )
    tracks = tp.filter_stubs(tracks, threshold=int(CFG["LENGTH_THRESHOLD"]))
    # Add split for compatibility with CME trackswhich need to be split currently
    tracks["split"] = 0
    tracks["file"] = save_name
    return tracks


@st.cache
def _track(videos, paths):
    """
    Particle tracking for one list of videos
    """
    detections = [
        _get_detections(
            v,
            diameter=int(CFG["TRACK_DIAMETER"]),
            min_mass_mult=float(CFG["TRACK_MINMASS_MULT"]),
        )
        for v in videos
    ]
    tracks = pd.concat(parmap.map(_link_df, zip(detections, paths)), sort=False)
    tracks.to_hdf(RESULTS_PATH, key="df")
    return tracks

def _is_a_peak(group):
    """
    Ensure that all tracks have at least start
    and end points that are lower than peak value
    """
    s = group["int_c1"]
    start = s[0]
    peak = s.max()
    end = s[1]

    is_peak = (peak > start) & (peak > end)
    return is_peak


if __name__ == "__main__":
    TIFFPATH = "data/kangmin_data/**/**/*.tif"
    RESULTS_PATH = "data/preprocessed/auxtracks-tpy.h5"
    CFG = ConfigObj("config/tracks_auxilin.cfg")

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

    if RERUN or not st._is_running_with_streamlit:
        try:
            print(
                "Files deleted! Rerunning...\n"
                "Use streamlit next time if you don't want to re-run"
            )
            os.remove(RESULTS_PATH)
        except FileNotFoundError:
            pass
        st.caching.clear_cache()

    try:
        tracks_c1 = _get_data(RESULTS_PATH)
    except FileNotFoundError:
        paths_c1 = _tiffpath(TIFFPATH).format("EGFP")
        paths_c1 = sorted(glob(paths_c1, recursive=True))
        videos_c1 = _get_videos(paths_c1)
        videos_c1 = _preprocess_videos(videos_c1, K = int(CFG["FFT_K"]), percentile=int(CFG["FFT_P"]))
        tracks_c1 = _track(videos_c1, paths_c1)

    # maintain compatibilit with dual channel script
    # no need to spend time tracking intensities in detail
    st.subheader("This script is meant for counting the auxilin tracks as a control\n"
                 "There's no need to do highly detailed intensity extraction here")
    tracks_c1.rename(columns = {"signal" : "int_c1"}, inplace = True)
    st.write(tracks_c1)

    n_videos = tracks_c1.groupby(["file"]).ngroups
    intensities = tracks_c1.groupby(["file", "particle"]).apply(lambda g: g["int_c1"]).ravel()
    st.subheader("Number of auxilin tracks detected: {} ({} videos)".format(len(intensities), n_videos))

    n_peaks = sum(tracks_c1.groupby(["file", "particle"]).apply(lambda g: _is_a_peak(g)))
    st.subheader("Number of auxilin tracks with real peaks: {}".format(n_peaks))

    nrows, ncols = 5, 5
    samples = lib.utils.sample_groups(tracks_c1, size = nrows * ncols, by = ["file", "particle"])
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols)
    ax = ax.ravel()
    for n, (_, group) in enumerate(samples.groupby(["file", "particle"])):
        ax[n].plot(group["int_c1"], color = "seagreen")
        ax[n].set_xticks(())
    plt.tight_layout()
    st.write(fig)

    fig, ax = plt.subplots()
    ax.hist(intensities, bins = 20)
    ax.set_title("Peak intensity distribution")
    ax.set_xlabel("Peak intensity")
    st.write(fig)

    _SET_CONFIG()


