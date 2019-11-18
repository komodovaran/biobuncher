import numpy as np
from msalign import msalign
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from lib.utils import sample_groups
from lib.math import resample_timeseries

np.random.seed(9)

X = np.load("results/intensities/tracks-cme_split-c1_res.npz")["data"]

zvals = np.row_stack(X[0:100, :, 1])

xvals = np.arange(0, zvals.shape[1], 1)
zvals_mean = np.median(zvals, axis = 0)

st.write(zvals.shape)
st.write(zvals_mean.shape)
peak_val = np.argmax(zvals_mean)

fig, ax = plt.subplots(nrows = 2, ncols = 2)
ax = ax.ravel()
ax[0].plot(zvals.T, alpha = 0.2, color = "black")
ax[1].plot(np.median(zvals.T, axis = 1))

reference_peaks = [peak_val]

peaks_aligned = msalign(
    xvals = xvals,
    zvals = zvals,
    peaks = reference_peaks,
    resolution = 100,
    grid_steps = 50,
    ratio = 2,
    iterations = 5,
    shift_range = [-300, 300],
    return_shifts = False,
    only_shift = True
)
ax[2].plot(peaks_aligned.T, alpha = 0.2, color = "black")
ax[3].plot(np.median(peaks_aligned.T, axis = 1))

st.write(fig)
# zvals_new = peaks()
#
# fig, ax = plt.subplots(nrows = 2)
# ax = ax.ravel()
#
# for s in zvals:
#     ax[0].plot(s)
#
# for s in zvals_new:
#     ax[1].plot(s)
#
# for a in ax:
#     a.set_xlim(6000)
#
# st.write(fig)
