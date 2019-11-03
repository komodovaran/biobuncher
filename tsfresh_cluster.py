import tsfresh
from tsfresh.feature_extraction import (
    MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters,
)
import numpy as np
import sklearn.model_selection
import streamlit as st
import lib.utils
import pandas as pd
import sklearn.decomposition
import sklearn.preprocessing
import plotly.express as px
import sklearn.cluster
import matplotlib.pyplot as plt

@st.cache
def _get_data():
    """
    Loads all traces and converts them to a padded tensor
    """
    X = np.load("results/2_intensities/2_intensities_padded_filtered.npz")["data"]
    return X


@st.cache
def _extract_ts_features(df):
    features = tsfresh.extract_features(
        timeseries_container=df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=ComprehensiveFCParameters(),
        n_jobs=8,
    )
    return features


def _remove_redundant_columns(df):
    """
    Removes columns with NaNs and all equal values
    """
    # Drop any column with NaNs
    df = df.dropna(how="any", axis=1)

    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(cols_to_drop, axis=1)
    return df


def _get_clusters(X, n_clusters=3):
    """Performs clustering and PCA for visualization"""
    # decompose first
    decomposer = sklearn.decomposition.PCA(n_components = 3)
    comps = decomposer.fit_transform(X)
    
    # cluster the decomposed
    clustering = sklearn.cluster.KMeans(n_clusters = n_clusters)
    c_label = clustering.fit_predict(comps)
    
    # stack value decomposition and
    comps = np.column_stack((comps, c_label))
    comps = pd.DataFrame(comps, columns=["PC1", "PC2", "PC3", "label"])
    return comps, c_label


if __name__ == "__main__":
    X = _get_data()
    X = X[..., [0]]
    st.write(X.shape)

    X = lib.utils.sample_max_normalize_3d(X, squeeze = False)
    df = lib.utils.ts_tensor_to_df(X)
    # df = lib.utils.ts_to_stationary(df, groupby = "id")
    st.subheader("timeseries df")
    st.write(df)

    st.subheader("Features extracted and redundant columns removed")
    features = _extract_ts_features(df)
    features = _remove_redundant_columns(features)
    st.write(features)
    st.write(features.columns)

    st.subheader("Standard scaled and PCA")
    scaler = sklearn.preprocessing.StandardScaler()
    features = scaler.fit_transform(features)
    
    n_clusters = 3
    pca_z, clusters = _get_clusters(features, n_clusters = n_clusters)
    st.write(pca_z.head())

    st.subheader("PCA of feature extraction")
    fig = px.scatter_3d(pca_z, x="PC1", y="PC2", z="PC3", color = "label")
    st.write(fig)
    
    for i in range(n_clusters):
        selected_idx, = np.where(clusters == i)
        selected_idx = selected_idx[0:9]  # take only for the number of plots shown
    
        st.subheader("Showing predictions for {}".format(i))
        fig, axes = plt.subplots(nrows = 3, ncols = 3)
        axes = axes.ravel()
        
        for i, ax in zip(selected_idx, axes):
            xi_true = lib.utils.remove_zero_padding(
                arr_true = X[i], arr_pred = None, padding = "before"
            )
            xi_true = np.squeeze(xi_true)
            
            ax.plot(xi_true)
            ax.plot([], [], label = "length: {}".format(len(xi_true)))
            ax.legend(loc = "upper right")
            ax.set_xticks(())
            ax.set_yticks(())
    
        plt.tight_layout()
        st.write(fig)
