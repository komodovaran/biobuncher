import tsfresh
from tsfresh.feature_extraction import (
    MinimalFCParameters,
    EfficientFCParameters,
    ComprehensiveFCParameters,
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
import lib.plotting

np.random.seed(1)


@st.cache
def _get_data():
    """
    Loads all traces
    """
    X = np.load("results/intensities/intensities_resampled_minlen15_relen50.npz")[
        "data"
    ]
    return X


@st.cache
def _extract_ts_features(df, fc_params, load_precomputed):
    savename = "results/extracted_features/tsfresh_features_{}.h5".format(
        fc_params
    )

    if load_precomputed:
        try:
            print("pre-computed extracted_features loaded from file")
            return pd.read_hdf(savename)
        except FileNotFoundError:
            return _extract_ts_features(df, fc_params, load_precomputed=False)

    if fc_params == "minimal":
        params = MinimalFCParameters()
    elif fc_params == "efficient":
        params = EfficientFCParameters()
    elif fc_params == "comprehensive":
        params = ComprehensiveFCParameters()
    else:
        raise ValueError("invalid parameters")

    for key in "length", "range_count", "count_above_mean", "count_below_mean":
        try:
            del params[key]
        except KeyError:
            pass

    features = tsfresh.extract_features(
        timeseries_container=df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=params,
        n_jobs=8,
    )

    features.to_hdf(savename, key="df")
    print("Saved features to disk")
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


def _get_clusters(X, n_clusters, n_components):
    """Performs clustering and PCA for visualization"""
    # Decompose data (1000+ dimensional) into something more managable
    decomposer = sklearn.decomposition.PCA(n_components=n_components)
    X_de = decomposer.fit_transform(X)

    st.write(X_de.shape)

    # cluster the decomposed
    clustering = sklearn.cluster.KMeans(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(X_de)

    # stack value decomposition and predicted labels
    X_de = np.column_stack((X_de, cluster_labels))
    X_de = pd.DataFrame(X_de)
    X_de["label"] = cluster_labels
    return X_de, cluster_labels


if __name__ == "__main__":
    st.subheader("Data shape")
    X_true = _get_data()
    st.write(X_true.shape)

    X = lib.utils.sample_max_normalize_3d(X_true, squeeze=False)
    df = lib.utils.ts_tensor_to_df(X)

    # remove zero padding
    df = df[df[0] != 0]
    # df = lib.utils.ts_to_stationary(df, groupby = "id")
    st.subheader("timeseries df")
    st.write(df.head())

    st.subheader("Features extracted and redundant columns removed")
    features = _extract_ts_features(
        df, fc_params="efficient", load_precomputed=True
    )
    features = _remove_redundant_columns(features)

    st.sidebar.subheader("Selected raw features")
    selected_columns = []
    if st.sidebar.checkbox("Channel 0", value = True):
        selected_columns += "0"
    if st.sidebar.checkbox("Channel 1", value = True):
        selected_columns += "1"
    if st.sidebar.checkbox("C0/C1 ratio", value = False):
        selected_columns += "2"
    if st.sidebar.checkbox("Steplength", value = False):
        selected_columns += "3"

    filtered_columns = [c for c in features.columns if str(c[0]) in selected_columns]
    features = features[filtered_columns]

    st.write(features.columns)

    st.subheader("Standard scaled and PCA")
    scaler = sklearn.preprocessing.StandardScaler()
    features = scaler.fit_transform(features)

    n_clusters = st.sidebar.slider(
        value=2, min_value=2, max_value=5, label="n clusters"
    )
    n_components = st.sidebar.slider(
        value=3,
        min_value=3,
        max_value=10,
        label="n components used for clustering",
    )
    pca_z, clusters = _get_clusters(
        features, n_clusters=n_clusters, n_components=n_components
    )
    st.write(pca_z.head())

    st.subheader("Decomposition of extracted extracted_features")
    fig = px.scatter_3d(pca_z.sample(n=200), x=0, y=1, z=2, color="label")
    st.write(fig)

    fig, ax = plt.subplots(nrows=n_clusters, figsize=(6, 10))
    for n in range(n_clusters):
        (selected_idx,) = np.where(clusters == n)
        mean_preds = np.mean(X_true[selected_idx], axis=0)
        ax[n].set_title(
            "fraction = {:.2f}".format(len(selected_idx) / len(X_true))
        )
        lib.plotting.plot_c0_c1_errors(
            mean_int_c0=mean_preds[:, 0], mean_int_c1=mean_preds[:, 1], ax=ax[n]
        )
    st.write(fig)

    for n in range(n_clusters):
        (selected_idx,) = np.where(clusters == n)
        # take only for the number of plots shown
        selected_idx = selected_idx[0:16]

        st.subheader("Showing predictions for {}".format(n))
        fig, axes = plt.subplots(nrows=4, ncols=4)
        axes = axes.ravel()

        mean_preds = []
        for i, ax in zip(selected_idx, axes):
            xi_true = lib.utils.remove_zero_padding(
                arr_true=X_true[i], arr_pred=None, padding="before"
            )
            xi_true = np.squeeze(xi_true)
            mean_preds.append(xi_true)
            lib.plotting.plot_c0_c1(
                int_c0=xi_true[:, 0], int_c1=xi_true[:, 1], ax=ax
            )
            ax.set_xticks(())
            ax.set_yticks(())
        plt.tight_layout()
        st.write(fig)
