import streamlit as st
import pandas as pd
import numpy as np

@st.cache
def _get_data():
    """
    Loads all traces and converts them to a padded tensor
    """
    df = pd.DataFrame(pd.read_hdf("results/intensities/tracks-cme_split-c1.h5"))
    return df

if __name__ == "__main__":
    df = _get_data()

    try:
        df["b"]
    except KeyError:
        df["b"] = 0

    st.write(df.head())
    df = df.groupby("b").filter(lambda x: len(x) > 20)

    st.subheader("examples")