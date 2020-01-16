import lib.config as c
import lib.utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

from lib.math import circle_mask, frame_roi_intensity, calc_steplength
from lib.utils import groupby_parallel_apply

if __name__ == "__main__":
    track_df = pd.read_csv(
        "data/preprocessed/A_CLTA-TagRFP EGFP-Aux1-A7D2_Cell1_1s_TagRFP_GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif.csv"
    )

    video_c0_path = "tom_data/A_CLTA-TagRFP EGFP-Aux1-A7D2/Cell1_1s/TagRFP/GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif"
    video_c1_path = "tom_data/A_CLTA-TagRFP EGFP-Aux1-A7D2/Cell1_1s/EGFP/GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_GFP  TIRF Q-1.tif"

    video_c0, video_c1 = [
        skimage.io.imread(path) for path in (video_c0_path, video_c1_path)
    ]
    if not video_c0.shape == video_c1.shape:
        raise ValueError("Videos are not equal shapes")

    video = np.stack((video_c0, video_c1), axis=-1)  # Stack with channels LAST
    print("Warning: Set to < 50 particles")
    track_df = track_df.query("particle < 50")
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

    def _get_intensities(group):
        """Function to run in parallel"""
        roi_intensities = np.zeros(shape=(len(group), dim * 2))
        for j, row in enumerate(group.itertuples()):
            y, x = row.y, row.x
            frame = video[row.frame]
            masks = circle_mask(
                yx=(y, x),
                indices=indices,
                gap_space=c.ROI_GAP,
                inner_area=c.ROI_INNER,
                outer_area=c.ROI_OUTER,
            )
            # Find intensities with ROIs from each channel separately
            # First 0, 2 is sig, then 1, 3 as bg
            roi_intensities[j, 0::dim], roi_intensities[j, 1::dim] = [
                frame_roi_intensity(frame[..., chan], *masks) for chan in range(dim)
            ]
        group[keys] = roi_intensities
        calc_steplength(group, "x", "y")
        return group

    df = groupby_parallel_apply(track_df.groupby("particle"), _get_intensities)

    fig, ax = plt.subplots(nrows=3)
    for _, grp in df.groupby("particle"):
        ax[0].plot(grp["int_c0"] - grp["bg_c0"])
        ax[1].plot(grp["signal"])
        ax[2].plot(grp["steplength"])

    ax[0].legend()
    ax[1].legend()
    plt.legend()
    plt.show()
