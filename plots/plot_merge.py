import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

files = glob("../results/tf_runs/merge/*.csv")

for f in files:
    merge = f.split("/")[-1].split("=")[3].split("_")[0]
    if merge == "ave":
        merge = "average"
    elif merge == "mul":
        merge = "multiply"
    elif merge == "sum":
        merge = "sum"
    else:
        merge = "single"

    df = pd.read_csv(f)
    mse = df["Value"]
    plt.plot(mse, label = merge)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

plt.savefig("../plots/merge.pdf")
plt.show()
