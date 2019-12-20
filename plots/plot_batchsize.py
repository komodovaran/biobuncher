import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

files = glob("results/tf_runs/batch_size/*.csv")

for f in files:
    df = pd.read_csv(f)
    act = f.split("/")[3].split("_")[4].split("=")[-1]
    mse = df["Value"]
    plt.plot(mse, label = act)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

plt.savefig("plots/activations.pdf")
