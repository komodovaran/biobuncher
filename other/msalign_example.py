import numpy as np
from msalign import msalign
import streamlit as st
import matplotlib.pyplot as plt

fname = r"examples/msalign_test_data.csv"
data = np.genfromtxt(fname, delimiter=",")
xvals = data[1:, 0]
zvals = data[1:, 1:].T

st.write(zvals.shape)

reference_peaks = [3991.4, 4598, 7964, 9160]

kwargs = dict(
    iterations=5,
    weights=[60, 100, 60, 100],
    resolution=100,
    grid_steps=20,
    ratio=2.5,
    shift_range=[-100, 100],
    )

@st.cache
def peaks():
    zvals_new = msalign(xvals, zvals, reference_peaks, **kwargs)
    return zvals_new

zvals_new = peaks()

fig, ax = plt.subplots(nrows = 2)
ax = ax.ravel()

for s in zvals:
    ax[0].plot(s)

for s in zvals_new:
    ax[1].plot(s)

for a in ax:
    a.set_xlim(6000)

st.write(fig)