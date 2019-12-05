import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

files = glob("../results/tf_runs/dimensionality/*.csv")
print(files)

for f in files:
    df = pd.read_csv(f)
    dim = f.split("dim=")[1][0:3].split("_")[0]
    mse = df["Value"]
    plt.plot(mse, label = "Encoded dimension = {}".format(dim))
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

plt.savefig("../plots/dimensionality.pdf")
plt.show()
