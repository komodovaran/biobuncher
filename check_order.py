import pandas as pd
import numpy as np
# import streamlit as st
# import matplotlib.pyplot as plt

BY = ["file", "particle"]
df = pd.DataFrame(pd.read_hdf("results/intensities/tracks-tpy_roi-int.h5"))
# X = np.load("results/intensities/tracks-tpy_roi-int_resampled-medianlen.npz")["data"]

grouped_df = df.groupby(BY)
groups = [group for _, group in grouped_df]

# df_samples = []
# for n, (_, group) in enumerate(grouped_df):
#     df_samples.append(group)
#     if n == 9:
#         break
# X_samples = X[0:10, ...]
#
# idx = 7
#
# fig, ax = plt.subplots(ncols = 2)
# ax[0].set_title("original")
# ax[0].plot(df_samples[idx]["int_c0"])
# ax[0].plot(df_samples[idx]["int_c1"])
# ax[1].set_title("resampled")
# ax[1].plot(X_samples[idx, :, 0])
# ax[1].plot(X_samples[idx, :, 1])
# st.write(fig)