import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import streamlit as st


def resample_timeseries(y, new_length = None):
    """
    Resamples timeseries by linear interpolation
    """
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

fig, ax = plt.subplots(nrows = 2)
X1 = np.random.exponential(2, 30)
X2 = np.random.normal(0, 1, 30)
X = np.column_stack((X1, X2))
y = resample_timeseries(y = X, new_length = 500)

ax[0].plot(X)
ax[1].plot(y)

st.write(X.shape)

st.write(fig)
