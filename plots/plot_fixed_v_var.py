import pandas as pd
import numpy as np
from lib.math import resample_timeseries
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import re
import scipy.signal
from lib.math import mean_squared_error as mse
X_train = np.load("../results/intensities/tracks-cme_var_traintest.npz", allow_pickle = True)["X_train"]

sns.set_style("darkgrid")

files = glob("../results/tf_runs/resampling/*")

for f in files:
    length = f.split("data=")[0].split("_")[-2]
    if length != "variable":
        length = "resampled"
    df = pd.read_csv(f)
    loss = df["Value"]
    plt.plot(loss, label = length)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
plt.savefig("length.pdf")

y = X_train[0]
x = np.arange(0, len(y), 1)
yfilt = scipy.signal.savgol_filter(y, axis = 0, window_length = 15, polyorder = 3)

y_re = resample_timeseries(y, new_length = 200)
yfilt_re = scipy.signal.savgol_filter(y_re, axis = 0, window_length = int(15 * (200 /len(y)))+1, polyorder = 3)

fig, ax = plt.subplots(nrows = 2, sharex = True)
mse0 = mse(y, yfilt, axis = (0, 1))
ax[0].set_title("variable length")
ax[0].plot(y, color = "salmon")
ax[0].plot(yfilt, color = "black")
ax[0].set_yticks(())
ax[0].plot([], color = "black", label = "mse: {:.2f}".format(mse0))
ax[0].legend(loc = "upper left")

mse1 = mse(y_re, yfilt_re, axis = (0, 1))
ax[1].set_title("resampled fixed length")
ax[1].plot(y_re, color = "salmon")
ax[1].plot(yfilt_re, color = "black")
ax[1].set_yticks(())
ax[1].plot([], color = "black", label = "mse: {:.2f}".format(mse1))
ax[1].legend(loc = "upper left")

plt.tight_layout()
plt.savefig("mse_resample.pdf")

plt.show()