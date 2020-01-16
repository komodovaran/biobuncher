import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import streamlit as st
import scipy.interpolate

np.random.seed(10)

def nd_fft(y):
    """
    Calculates fast fourier transform for each feature in a ND-timeseries
    """
    y = y.reshape(len(y), -1)
    # y = y - np.mean(y) # center the signal to avoid zero component
    ndim = y.shape[1]

    new_y = np.zeros((y.shape[0] // 2, ndim))
    for i in range(ndim):
        yf = fft(y[:, i])

        # Remove mirrored and negative parts
        yf_single = np.abs(yf[:len(y)//2])
        new_y[:, i] = yf_single
    return new_y

def resample_timeseries(y, new_length = None):
    """
    Resamples timeseries by linear interpolation
    """
    y = y.reshape(len(y), -1)
    ndim = y.shape[1]

    x = range(len(y))
    if new_length is None:
        new_length = len(x)

    new_x = np.linspace(min(x), max(x), new_length)
    new_y = np.zeros((new_length, ndim))

    # interpolate for each of the channels individually and collect
    for i in range(ndim):
        f = scipy.interpolate.interp1d(x, y[:, i])
        new_y[:, i] = f(new_x)
    return new_y

fig, ax = plt.subplots(nrows = 2, ncols = 4)
ax = ax.ravel()

final_len = 50

peak = np.array([0, 1, 2, 3, 5, 8, 12, 5, 2, 1, 0, 1, 0, 0, 0, 0]) ** 2
peak = resample_timeseries(peak, 200).ravel()
peak += np.random.normal(0, 10, len(peak))
peak = resample_timeseries(peak, final_len)

zeros = np.zeros(peak.shape)

y1 = np.concatenate((zeros, peak))
y1 = resample_timeseries(y1, final_len)

y2 = np.concatenate((peak, zeros))
y2 = resample_timeseries(y2, final_len)

y3 = y1 + y2
y4 = y1 + y2 * 2

ff_ = [nd_fft(y) for y in (y1, y2, y3, y4)]
ff1, ff2, ff3, ff4 = [ff/np.max(ff) for ff in ff_]

st.subheader("Zero-padded to match max-length")
ax[0].plot(y1, color = "salmon")
ax[0].set_ylabel("Time domain")
ax[1].plot(y2, color = "royalblue")
ax[2].plot(y3, color = "seagreen")
ax[3].plot(y4, color = "orange")

ax[4].plot(ff1, color = "salmon")
ax[4].set_ylabel("Frequency domain")
ax[5].plot(ff2, color = "royalblue")
ax[6].plot(ff3, color = "seagreen")
ax[7].plot(ff4, color = "orange")

for a in ax:
    a.set_xticks(())
    a.set_yticks(())
plt.tight_layout()
st.write(fig)