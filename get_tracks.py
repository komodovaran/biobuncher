from glob import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage.color
import skimage.exposure
import skimage.io
import sklearn.preprocessing
import streamlit as st
import trackpy as tp

import lib.math
import lib.plotting
import lib.utils

from configobj import ConfigObj

sns.set(context="notebook", style="darkgrid", palette="muted")
tp.quiet()


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


def _find_tracks(path):
    """
    Set tracking parameters through tests
    """
    save_name = remove_parent_dir(path, 1)
    video = skimage.io.imread(path)
    features = tp.batch(
        frames=video,
        diameter=CONFIG["TRACK_DIAMETER"],
        minmass=np.mean(video) * CONFIG["TRACK_MINMASS_MULT"],
        engine="numba",
    )

    # memory keeps track of particle for a number of frames if mass is below cutoff
    tracks = tp.link_df(
        features,
        search_range=CONFIG["TRACK_RANGE"],
        memory=CONFIG["TRACK_MEMORY"],
    )
    tracks = tp.filter_stubs(tracks, threshold=CONFIG["LENGTH_THRESHOLD"])
    tracks["file"] = save_name

    # for compatibility with CME tracks which need to be split
    tracks["split"] = 0
    return tracks


def _track(video_paths):
    """
    Do the particle tracking
    """
    with Pool(cpu_count()) as p:
        tracks = pd.concat(p.map(_find_tracks, video_paths), sort=False)
    tracks.to_hdf(RESULTS_PATH, key="df")
    return tracks


def _median_filter_sort(group):
    s = group["int_c1"]
    s = sklearn.preprocessing.minmax_scale(s)
    med = np.median(s)
    std = np.std(s)
    if np.any(s) > med + 3 * std:
        return group
    else:
        return None


def _get_intensities(group):
    """Function to run in parallel"""
    roi_intensities = np.zeros(shape=(len(group), dim * 2))

    lib.math.calc_steplength(group, "x", "y")

    for j, row in enumerate(group.itertuples()):
        y, x = row.y, row.x
        frame = video[row.frame]
        # Find intensities with ROIs from each channel separately
        # First 0, 2 is sig, then 1, 3 as bg
        masks = lib.math.circle_mask(
            yx=(y, x),
            indices=indices,
            gap_space=CONFIG["ROI_GAP"],
            inner_area=CONFIG["ROI_INNER"],
            outer_area=CONFIG["ROI_OUTER"],
        )
        for chan, idx in enumerate(lib.utils.pairwise(range(dim * 2))):
            roi_intensities[j, idx] = lib.math.frame_roi_intensity(
                frame[..., chan], *masks
            )
    group[keys] = roi_intensities
    return group


def _SET_CONFIG():
    """
    Writes configs to a file, so they can be reloaded on next streamlit
    instance. They must be loaded as floats!
    """
    CONFIG["TRACK_MINMASS_MULT"] = TRACK_MINMASS_MULT
    CONFIG["TRACK_MEMORY"] = TRACK_MEMORY
    CONFIG["TRACK_DIAMETER"] = TRACK_DIAMETER
    CONFIG["TRACK_MEMORY"] = TRACK_MEMORY
    CONFIG["TRACK_RANGE"] = TRACK_RANGE
    CONFIG["LENGTH_THRESHOLD"] = LENGTH_THRESHOLD
    CONFIG["ROI_INNER"] = ROI_INNER
    CONFIG["ROI_OUTER"] = ROI_OUTER
    CONFIG["ROI_GAP"] = ROI_GAP
    CONFIG.write()


if __name__ == "__main__":
    TIFFPATH = "data/kangmin_data/**/**/*.tif"
    RESULTS_PATH = "results/intensities/tracks-tpy.h5"
    CONFIG = ConfigObj("config/get_tracks.cfg")

    # Setup for the app
    TRACK_MINMASS_MULT = st.sidebar.number_input(
        value=float(CONFIG["TRACK_MINMASS_MULT"]),
        label="track minmass multiplier",
    )

    TRACK_DIAMETER = st.sidebar.number_input(
        value=int(CONFIG["TRACK_DIAMETER"]),
        label="Track Spot Diameter (must be odd)",
    )
    TRACK_MEMORY = st.sidebar.number_input(
        value=int(CONFIG["TRACK_MEMORY"]), label="Track Memory"
    )
    TRACK_RANGE = st.sidebar.number_input(
        value=int(CONFIG["TRACK_RANGE"]), label="Track Search Range"
    )
    LENGTH_THRESHOLD = st.sidebar.number_input(
        value=int(CONFIG["LENGTH_THRESHOLD"]), label="Track Min Length"
    )
    ROI_INNER = st.sidebar.number_input(
        value=int(CONFIG["ROI_INNER"]), label="ROI Inner"
    )
    ROI_OUTER = st.sidebar.number_input(
        value=int(CONFIG["ROI_OUTER"]), label="ROI Outer"
    )
    ROI_GAP = st.sidebar.number_input(
        value=int(CONFIG["ROI_GAP"]), label="ROI Gap"
    )

    TRACK_BUTTON = st.sidebar.button(label="Re-run particle tracking (SLOW!)")

    # Actual computations
    paths_egfp, paths_tagrfp = [
        _tiffpath(TIFFPATH).format(s) for s in ("EGFP", "TagRFP")
    ]
    tiffs_egfp, tiffs_tagrfp = [
        sorted(glob(paths, recursive=True))
        for paths in (paths_egfp, paths_tagrfp)
    ]
    if len(tiffs_egfp) != len(tiffs_tagrfp):
        raise ValueError("Not enough videos for master/slave tracking")

    if TRACK_BUTTON or not st._is_running_with_streamlit:
        _SET_CONFIG()
        tracks = _track(tiffs_tagrfp)
    else:
        try:
            tracks = pd.DataFrame(pd.read_hdf((RESULTS_PATH)))
        except FileNotFoundError:
            tracks = _track(tiffs_tagrfp)

    intensity_savepath = RESULTS_PATH[:-3] + "_roi-int.npz"
    if TRACK_BUTTON or not (st._is_running_with_streamlit):
        track_files = tracks["file"].unique()
        intensities = []
        for egfp_n, tagrfp_n, tracks_n in zip(
            tiffs_egfp, tiffs_tagrfp, track_files
        ):
            ti = tracks[tracks["file"] == tracks_n]
            video_c0, video_c1 = [
                skimage.io.imread(path) for path in (tagrfp_n, egfp_n)
            ]
            if not video_c0.shape == video_c1.shape:
                raise ValueError("Videos are not equal shapes")

            video = np.stack(
                (video_c0, video_c1), axis=-1
            )  # Stack with channels LAST
            indices = np.indices(video[0, ..., 0].shape)

            # Initialize needed columns
            for i in range(video.shape[-1]):
                ti["int_c{}".format(i)] = 0
                ti["bg_c{}".format(i)] = 0

            dim = video.shape[-1]
            # Keys for correct number of int and bg channels
            keys = lib.utils.flatten_list(
                [("int_c{}".format(c), "bg_c{}".format(c)) for c in range(dim)]
            )
            # Initialize
            ti[keys] = 0

            intensities.append(
                lib.utils.groupby_parallel_apply(
                    ti.groupby("particle"), _get_intensities
                )
            )

        intensity_df = pd.concat(intensities)
        intensity_df.to_hdf(intensity_savepath, key="df")
    else:
        intensity_df = pd.read_hdf(intensity_savepath)

    total_groups = intensity_df.groupby(["file", "particle"]).ngroups

    st.subheader("Total number of tracks")
    st.write(total_groups)

    has_aux = lib.utils.groupby_parallel_apply(
        intensity_df.groupby(["file", "particle"]), _median_filter_sort
    )

    st.subheader("Has auxilin")
    st.write(has_aux.groupby((["file", "particle"])).ngroups)

    st.subheader("Example tracks")

    nrows, ncols = 6, 5
    samples = []
    n = 0
    for _, group in intensity_df.groupby(["file", "particle"]):
        if n == nrows * ncols:
            break
        samples.append(group)
        n += 1
    samples = pd.concat(samples)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    ax = ax.ravel()
    for n, (_, group) in enumerate(samples.groupby(["file", "particle"])):
        lib.plotting.plot_c0_c1(
            ax=ax[n], int_c0=group["int_c0"], int_c1=group["int_c1"]
        )
        ax[n].axhline(np.median(group["int_c1"]), ls="--", color="black")
        ax[n].set_xticks(())
    plt.tight_layout()
    st.write(fig)
