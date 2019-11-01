import numpy as np
import skimage.draw
import skimage.exposure
import skimage.feature
import skimage.color
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.colors
from tqdm import tqdm
import pandas as pd
import lib.math
import lib.config as c


def get_cmaps():
    mask_cmap = plt.cm.Reds
    mask_cmap.set_under("k", alpha=0)

    mask_cmap_bg = plt.cm.Blues
    mask_cmap_bg.set_under("k", alpha=0)

    return mask_cmap, mask_cmap_bg


video_path = "../data/A_CLTA-TagRFP EGFP-Aux1-A7D2/Cell1_1s/TagRFP/GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif"
video = skimage.io.imread(video_path)

track_path = "../results/_A_CLTA-TagRFP EGFP-Aux1-A7D2_Cell1_1s_TagRFP_GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif.csv"
tracks = pd.read_csv(track_path)

frame_n = 0
frame = video[frame_n]

tracks_n = tracks.query("frame == 0")
indices = np.indices(frame.shape)
ctr_masks = np.zeros(frame.shape)
bg_masks = np.zeros(frame.shape)
for i in tracks_n.itertuples():
    x, y = i.x, i.y
    ctr_mask, bg_mask = lib.math.circle_mask(
        yx=(y, x),
        inner_area=c.ROI_INNER,
        outer_area=c.ROI_OUTER,
        gap_space=c.ROI_GAP,
        indices=indices,
    )
    roi_intensity, bg_intensity = lib.math.roi_intensity(
        frame, roi_mask=ctr_mask, bg_mask=bg_mask
    )

    ctr_masks += ctr_mask
    bg_masks += bg_mask

ctr_cmap, bg_cmap = get_cmaps()

plt.imshow(skimage.exposure.rescale_intensity(frame), cmap = "Greys_r")
plt.imshow(ctr_masks, cmap = ctr_cmap, clim = (0.5, 1), alpha = 0.6)
plt.imshow(bg_masks, cmap = bg_cmap, clim = (0.5, 1), alpha = 0.6)
plt.show()
