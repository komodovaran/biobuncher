import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure
import skimage.io
import trackpy as tp

import lib.config as c
import lib.math


def _get_cmaps():
    mask_cmap = plt.cm.Blues
    mask_cmap.set_under("k", alpha=0)

    mask_cmap_bg = plt.cm.Blues
    mask_cmap_bg.set_under("k", alpha=0)

    return mask_cmap, mask_cmap_bg

def _single_frame(video, frame):
    return np.expand_dims(video[frame, ...], axis = 0)

video_path = "../data/kangmin_data/B_CLTA-TagRFP EGFP-Aux1 EGFP-GAK F6-1/Cell1_1s/TagRFP/GFP-ND20 Exp100ms RFP-ND12 Exp100ms - 1_Cy3  TIRF Q-1.tif"
video = skimage.io.imread(video_path)

frame_n = 20

tracks = tp.batch(
    frames = _single_frame(video, frame_n),
    diameter = 7,
    minmass = np.mean(video) * 1.5,
    engine = "numba",
    processes = 8,
)

frame = video[frame_n]

indices = np.indices(frame.shape)
ctr_masks = np.zeros(frame.shape)
bg_masks = np.zeros(frame.shape)
for i in tracks.itertuples():
    x, y = i.x, i.y
    ctr_mask, bg_mask = lib.math.circle_mask(
        yx=(y, x),
        inner_area=c.ROI_INNER,
        outer_area=c.ROI_OUTER,
        gap_space=c.ROI_GAP,
        indices=indices,
    )
    roi_intensity, bg_intensity = lib.math.frame_roi_intensity(
        frame, roi_mask=ctr_mask, bg_mask=bg_mask
    )

    ctr_masks += ctr_mask
    bg_masks += bg_mask

ctr_cmap, bg_cmap = _get_cmaps()

plt.imshow(skimage.exposure.rescale_intensity(frame), cmap = "Greys_r")
# plt.imshow(ctr_masks, cmap = ctr_cmap, clim = (0.5, 1), alpha = 0.3)
plt.imshow(bg_masks, cmap = bg_cmap, clim = (0.5, 1), alpha = 0.3)
plt.show()
