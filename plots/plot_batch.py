import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import datetime
fmt = mdates.DateFormatter('%s')

sns.set_style("darkgrid")

files = sorted(glob("../results/tf_runs/batch_size/*.csv"))

fig, ax = plt.subplots(nrows = 1, ncols = 1)

for f in files:
    df = pd.read_csv(f)
    dim = f.split("bat=")[1].split("_")[0]
    mse = df["Value"]
    time = df["Wall time"]
    time = time - time[0]
    seconds = time
    hours = time/60/60
    ax.plot(hours, mse, label = "Batch size = {}".format(dim))
    plt.legend()
    plt.xlabel("time (hours)")
    plt.ylabel("MSE")

plt.savefig("../plots/batch_size.pdf")
plt.show()
