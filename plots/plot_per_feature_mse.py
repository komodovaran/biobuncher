import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import datetime
import numpy as np

sns.set_style("darkgrid")

files = sorted(glob("../results/tf_runs/mse_per_feature/*.npz"))
names = "both", "0", "1"

fig, ax = plt.subplots(nrows = 2)
ax = ax.ravel()

bins = np.arange(0, 3, 0.025)
data = []
for (f, n) in zip(files, names):
    data.append(np.load(f)["data"])

p = dict(bins = bins, alpha = 0.5, density = True)

ax[0].set_title("Combined model")
ax[0].hist(data[0][:, 0], **p, label = "feature 0")
ax[0].hist(data[0][:, 1], **p, label = "feature 1")

ax[1].set_title("Separate models")
ax[1].hist(data[1], **p, label = "feature 0")
ax[1].hist(data[2], **p, label = "feature 1")

for a in ax:
    a.set_ylabel("Probability Density")
    a.set_xlabel("MSE")
    a.legend(loc = "upper right")
    a.set_xlim(0, 2)
plt.tight_layout()
plt.savefig("../plots/mse_features.pdf")
plt.show()