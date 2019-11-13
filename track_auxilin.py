from glob import glob

import streamlit as st
from configobj import ConfigObj

import lib.math
import lib.plotting
import lib.utils
from get_tracks import _get_videos, _tiffpath, _preprocess_videos
from lib.utils import timeit

@timeit
def _preprocess_videos(videolist):
    """Subtracts bg from a list of videos"""
    videos = [lib.math.fft_bg_video(video = v, K=int(CFG["FFT_K"]), percentile=int(CFG["FFT_P"])) for v in videolist]
    return videos

def _SET_CONFIG():
    """
    Writes configs to a file, so they can be reloaded on next streamlit
    instance. They must be loaded as floats!
    """
    CFG["FFT_K"] = FFT_K
    CFG["FFT_P"] = FFT_P
    CFG.write()

if __name__ == "__main__":
    TIFFPATH = "data/kangmin_data/**/**/*.tif"
    RESULTS_PATH = "results/intensities/auxtracks-tpy.h5"
    CFG = ConfigObj("config/get_tracks.cfg")

    FFT_K = st.sidebar.slider(
        min_value = 1, max_value = 30, value = int(CFG["FFT_K"]), label = "FFT wiggle"
    )
    FFT_P = st.sidebar.slider(
        min_value = 1, max_value = 100, value = int(CFG["FFT_P"]), label = "FFT noise"
    )

    st.sidebar.subheader("This is for tracking auxilin only, as a single-track control")

    paths_c1 = _tiffpath(TIFFPATH).format("EGFP")
    paths_c1 = sorted(glob(paths_c1, recursive = True))
    videos_c1 = _get_videos(paths_c1)
    videos_c1 = _preprocess_videos(videos_c1)