import pandas as pd
import numpy as np
import streamlit as st
import os.path
import matplotlib.pyplot as plt
np.random.seed(0)

@st.cache
def get_data():
    path = "results/intensities/tracks-CLTA-TagRFP EGFP-Aux1-A7D2 EGFP-Gak-F6_filt5_var.h5"
    df = pd.DataFrame(pd.read_hdf(path))

    # df = pd.DataFrame({"a"     : np.random.normal(0, 100, 1000000),
    #                    "index1": np.random.randint(0, 200, 1000000),
    #                    "index2": np.random.randint(0, 200, 1000000)})
    return df

@st.cache
def get_group(df, n):
    grouped_df = df.groupby(["file", "particle"])
    keys = list(grouped_df.groups.keys())
    group = grouped_df.get_group(keys[n])
    return group

n = st.number_input(min_value = 0, max_value = 100, label = "ID of group to fetch")

df = get_data()
group = get_group(df, n = n)

pos = st.number_input(min_value = 0, max_value = len(group), label = "Multiply fetched group or something")

fig, ax = plt.subplots()
ax.plot(range(len(group)), group["int_c0"])
ax.axvline(pos)
st.write(fig)

st.write(group)