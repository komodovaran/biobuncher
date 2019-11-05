import os.path
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import seaborn as sns
import skimage.color
import skimage.exposure
import skimage.io
import trackpy as tp

import lib.config as c

sns.set(context = "notebook", style = "darkgrid", palette = "muted")
tp.quiet()

def _find_tracks(path):
    """
    Set tracking parameters through tests
    """
    save_name = path.lstrip("data/").replace("/", "_")
    video = skimage.io.imread(path)
    features = tp.batch(
        frames = video,
        diameter = c.TRACK_DIAMETER,
        minmass = np.mean(video) * c.TRACK_MINMASS_MULT,
        engine = "numba",
    )

    # memory keeps track of particle for a number of frames if mass is below cutoff
    tracks = tp.link_df(features, search_range = 1, memory = 1)
    tracks = tp.filter_stubs(tracks, threshold = 15)
    tracks.to_csv(os.path.join("results/tracks/", save_name + ".csv"))

if __name__ == "__main__":
    paths = sorted(glob("data/**/TagRFP/*.tif", recursive = True))

    with Pool(cpu_count()) as p:
        p.map(_find_tracks, paths)