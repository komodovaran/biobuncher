import pandas as pd
import numpy as np

df = {"pos_x": [0, 2, 4, 6, 8, 10],
      "pos_y": [0, 2, 4, 6, 8, 10]}
df = pd.DataFrame(df)

df[["diff_x", "diff_y"]] = df.rolling(window = 2).apply(lambda row: row[1] - row[0], raw = True) # because it returns sub-df with size = window
df["steplength"] = np.sqrt(df["diff_x"]**2 + df["diff_y"]**2)