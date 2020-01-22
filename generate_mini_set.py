import pandas as pd

df = pd.DataFrame(pd.read_hdf("data/preprocessed/combined_filt5_var.h5"))

# Take the first 100 of every group
df_mini = df[df["sub_id"] <= 100]

df_mini.to_hdf("data/preprocessed/mini_filt5_var.h5", key = "df")