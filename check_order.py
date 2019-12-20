import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from lib.tfcustom import VariableTimeseriesBatchGenerator
import lib.utils

@st.cache
def load_df(df_path):
    df = pd.DataFrame(pd.read_hdf(df_path))
    return df

if __name__ == "__main__":
    BY = ["file", "particle"]

    df_path = "results/intensities/tracks-CLTA-TagRFP EGFP-Gak-A8_var.h5"

    df = load_df(df_path)

    grouped_df = df.groupby(BY)
    keys = list(grouped_df.groups.keys())

    find = 6328

    single = grouped_df.get_group(keys[find])

    plt.plot(single[["int_c0", "int_c1"]].values)
    plt.show()