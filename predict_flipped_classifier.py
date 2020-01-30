import numpy as np
import tensorflow as tf
import tensorflow.keras.models
from tqdm import tqdm

from lib.math import f1_m
from lib.tfcustom import VariableTimeseriesBatchGenerator
import matplotlib.pyplot as plt
from st_predict import _pca
import umap.umap_ as umap
import streamlit as st

autoencoder = tensorflow.keras.models.load_model(
    "models/20200129-2039_lstm_classifier_data=combined_filt20_var.npz_dim=128_bat=4/model_008.h5", custom_objects = {"f1_m" : f1_m})

X_true = np.load("data/preprocessed/combined_filt20_var.npz", allow_pickle = True)["data"]

X_true = X_true[0:10000]

indices = np.arange(len(X_true))
n_features = X_true[0].shape[-1]

encoder = tensorflow.keras.models.Model(
                inputs=autoencoder.input,
                outputs=autoencoder.get_layer("bidirectional_1").output,
            )

X_ = VariableTimeseriesBatchGenerator(
    X = X_true.tolist(),
    indices = indices,
    max_batch_size = 512,
    shuffle_samples = True,
    shuffle_batches = True,
)

indices = X_.indices

X_ = tf.data.Dataset.from_generator(
    generator = X_,
    output_types = (tf.float64, tf.float64),
    output_shapes = ((None, None, n_features), (None, None, n_features),),
)

# Predict on batches and unravel for single-item use
X_true, X_pred, features, mse = [], [], [], []
for xi_true, _ in tqdm(X_):
    _xi_true = xi_true
    fi = encoder.predict_on_batch(_xi_true)

    X_true.extend(np.array(xi_true))
    features.extend(np.array(fi))

X_true, features = map(
    np.array, (X_true, features)
)

pca, _ = _pca(features, embed_into_n_components = 2)


u = umap.UMAP(
        n_components = 2,
        random_state = 42,
        n_neighbors = 100,
        min_dist = 0.0,
        init = "spectral",
        verbose = True,
    )
e = u.fit_transform(features)

plt.plot(e[:, 0], e[:, 1], "o")
plt.show()