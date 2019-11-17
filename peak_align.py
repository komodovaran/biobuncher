import numpy as np
from msalign import msalign
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from lib.utils import sample_groups
from lib.math import resample_timeseries

np.random.seed(9)

data = pd.read_hdf("results/intensities/tracks-cme_split-c1.h5")
samples = sample_groups(data, 3, by=["file", "particle"])
samples = [group for _, group in samples.groupby(["file", "particle"])]

st.write(samples)

zvals = np.row_stack(
    [resample_timeseries(sample[["int_c0", "int_c1"]]/sample[["int_c0", "int_c1"]].max(), 50) for sample in samples]
)


xvals = np.arange(0, zvals.shape[1], 1)
st.write(zvals.shape)

fig, ax = plt.subplots(nrows=2, ncols = 2)
ax = ax.ravel()
ax[0].plot(zvals.T)
ax[1].plot(zvals.T.mean(axis = 1))

reference_peaks = [20]

peaks_aligned, shifts = msalign(
    xvals=xvals,
    zvals=zvals,
    peaks=reference_peaks,
    resolution=100,
    grid_steps=50,
    ratio=2,
    iterations=5,
    shift_range=[-100, 100],
    return_shifts = True
)
st.write(shifts)

ax[2].plot(peaks_aligned.T)
ax[3].plot(peaks_aligned.T.mean(axis = 1))

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
