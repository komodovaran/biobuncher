import os.path
from glob import glob

import numpy as np
import pandas as pd
import skimage.io
from tqdm import tqdm

import lib.config as C
import lib.utils
from lib.math import calc_steplength, circle_mask, frame_roi_intensity
from lib.utils import groupby_parallel_apply, pairwise


def _get_paths():
    """
    Obtain all the paths
    """
    # Primary channel is TagRFP
    video_c0_paths, video_c1_paths = [
        glob("tom_data/**/{}/*.tif".format(c), recursive=True) for c in ("TagRFP", "EGFP")
    ]
    video_c0_paths, video_c1_paths = [sorted(p) for p in (video_c0_paths, video_c1_paths)]

    # Collect all paths first (makes parallel runs later easier)
    path_list = []
    for path0, path1 in zip(video_c0_paths, video_c1_paths):
        track_path = os.path.join(
            "results/tracks/", path0.lstrip("tom_data/").replace("/", "_") + ".csv"
        )
        results_path = track_path.replace("tracks", "intensities")
        path_list.append((path0, path1, track_path, results_path))
    return path_list


def _get_intensities(group):
    """Function to run in parallel"""
    roi_intensities = np.zeros(shape=(len(group), dim * 2))

    calc_steplength(group, "x", "y")

    for j, row in enumerate(group.itertuples()):
        y, x = row.y, row.x
        frame = video[row.frame]
        # Find intensities with ROIs from each channel separately
        # First 0, 2 is sig, then 1, 3 as bg
        masks = circle_mask(
            yx=(y, x),
            indices=indices,
            gap_space=C.ROI_GAP,
            inner_area=C.ROI_INNER,
            outer_area=C.ROI_OUTER,
        )
        for chan, idx in enumerate(pairwise(range(dim * 2))):
            roi_intensities[j, idx] = frame_roi_intensity(frame[..., chan], *masks)

    group[keys] = roi_intensities
    return group


if __name__ == "__main__":
    path_list = _get_paths()

    # Loop over each file with appropriate paths set and find intensities
    for idx, (TAGRFP_path, EGFP_path, track_path, results_path) in tqdm(enumerate(path_list)):
        track_df = pd.read_csv(track_path)
        video_c0, video_c1 = [skimage.io.imread(path) for path in (TAGRFP_path, EGFP_path)]
        if not video_c0.shape == video_c1.shape:
            raise ValueError("Videos are not equal shapes")

        video = np.stack((video_c0, video_c1), axis=-1)  # Stack with channels LAST
        indices = np.indices(video[0, ..., 0].shape)

        # Initialize needed columns
        for i in range(video.shape[-1]):
            track_df["int_c{}".format(i)] = 0
            track_df["bg_c{}".format(i)] = 0

        dim = video.shape[-1]
        # Keys for correct number of int and bg channels
        keys = lib.utils.flatten_list(
            [("int_c{}".format(c), "bg_c{}".format(c)) for c in range(dim)]
        )
        # Initialize
        track_df[keys] = 0

        df = groupby_parallel_apply(track_df.groupby("particle"), _get_intensities)
        df.to_csv(results_path)
