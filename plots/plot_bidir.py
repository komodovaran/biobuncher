import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

files = glob("../results/tf_runs/bidirectional/*.csv")

for f in files:
    if f.split("/")[-1].split("_")[0] == "single":
        direction = "single"
    else:
        direction = "bidirectional"

    df = pd.read_csv(f)
    mse = df["Value"]
    plt.plot(mse, label = direction)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

plt.savefig("../plots/bidirectional.pdf")
plt.show()
