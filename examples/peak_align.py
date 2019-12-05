import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from msalign import msalign

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