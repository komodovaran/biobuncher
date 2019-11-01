import pandas as pd
from glob import glob
import os.path

directories = glob("results/*")
for d in directories:
    savename = os.path.join(d, d.split("/")[-1] + ".h5")
    files = glob(os.path.join(d, "*.csv"))
    if not files:
        continue

    dfs = []
    for f in files:
        sub = pd.read_csv(f)
        sub["file"] = f
        dfs.append(sub)
    df = pd.concat(dfs)
    df.to_hdf(path_or_buf = savename, key = "df")

# Try to load, to check if it works correctly
df = pd.read_hdf("results/1_tracks/1_tracks.h5", key = "df")
