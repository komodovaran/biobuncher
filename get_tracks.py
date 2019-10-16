import matplotlib.pyplot as plt
import numpy as np
import pims
import trackpy as tp
import seaborn as sns
import pandas as pd
import skimage.io
import skimage.color
import skimage.exposure
import os.path
import lib.config as c
from glob import glob

sns.set(context = "notebook", style = "darkgrid", palette = "muted")
tp.quiet()

def eval_frame(arr, frame_num):
    """Evaluates a given frame for tracking parameters"""
    frame = arr[frame_num, ...]
    
    located_features = tp.locate(raw_image = frame, diameter = c.TRACK_DIAMETER, minmass = c.TRACK_MINMASS_MULT * frame.mean())
    located_features.head()  # shows the first few rows of data
    
    fig1, axes1 = plt.subplots(ncols = 2)
    axes1 = axes1.ravel()
    
    axes1[0].imshow(skimage.exposure.equalize_hist(frame))
    tp.annotate(located_features, frame, ax = axes1[0], plot_style = {"alpha": 0.2})
    axes1[1].hist(located_features["mass"], bins = 20)
    axes1[1].set(xlabel = "mass", ylabel = " count")
    
    fig2 = plt.figure()
    tp.subpx_bias(located_features)


def plot_tracks(track_df, max_plots):
    fig, axes = plt.subplots(nrows = max_plots // 2, ncols = 2)
    axes = axes.ravel()
    for i, (n, particle) in enumerate(track_df.groupby("particle")):
        if i == max_plots:
            break
        axes[i].plot(particle["signal"])
    plt.tight_layout()


if __name__ == "__main__":
    path = "data/A_CLTA-TagRFP EGFP-Aux1-A7D2/Cell1_1s/TagRFP/GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif"
    savename = path.replace("/", "_").lstrip("data/")
    
    video = skimage.io.imread(path)
    
    mean_int = video.mean()
    features = tp.batch(video, diameter = c.TRACK_DIAMETER, minmass = mean_int * c.TRACK_MINMASS_MULT, engine = "numba")
    
    # memory keeps track of particle for a number of frames if mass is below cutoff
    tracks = tp.link_df(features, search_range = 3, memory = 1)
    tracks = tp.filter_stubs(tracks, threshold = 3)
    tracks.to_csv(os.path.join("results", savename + ".csv"))

    # plot_tracks(track_df = tracks, max_plots = 20)
    # plt.show()