import pandas as pd
from glob import glob
import os.path

def _CSV_TO_H5():
    """Finds and converts CSV files to .h5 for faster loading"""
    directories = glob("results/*")
    print(directories)
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

_CSV_TO_H5()