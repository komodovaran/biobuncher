import pandas as pd
import numpy as np

df = {"pos_x": [0, 2, 4, 6, 8, 10],
      "pos_y": [0, 2, 4, 6, 8, 10],
      "id"   : [1, 1, 1, 2, 2, 2]}
df = pd.DataFrame(df)

for name, group in df.groupby("id"):
    group[["dif_x", "dif_y"]] = group[["pos_x", "pos_y"]].rolling(window = 2).apply(lambda row: row[1] - row[0],
                                                                                    raw = True)
    group["len"] = np.sqrt(group["dif_x"] ** 2 + group["dif_y"] ** 2)

print(df)