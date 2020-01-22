import os.path
import pandas as pd
import lib.globals

file = "data/preprocessed/mini_filt5_var.h5"

df = pd.DataFrame(pd.read_hdf(file))
original_columns = df.columns

if "source" not in df.columns:
    df["source"] = os.path.basename(file)

if "id" not in df.columns:
    df["id"] = df.groupby(lib.globals.groupby).ngroup()

fixed_columns = df.columns

# Only overwrite df if anything was changed
if original_columns != fixed_columns:
    df.to_hdf(file, key = "df")