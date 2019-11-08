import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np

@st.cache
def _get_data(path):
    return pd.DataFrame(pd.read_hdf(path))

if __name__ == "__main__":
    PATH = "results/intensities/tracks-cme_split-both.h5"

    df = _get_data(PATH)
    cat = df.groupby(["file", "particle"])["catIdx__"].first().values

    fig, ax = plt.subplots()
    ax.hist(cat)

    st.subheader("Distribution of track categories\n1-4 is good, 5-8 is potentially bad")
    st.write(fig)

    st.subheader("Number of NaN values:")
    nans = df.isna().sum()
    st.write(nans)