import matplotlib.pyplot as plt

from st_predict import _umap_embedding, _pca, _latest_model, _get_encoding_layer, _predict
from tensorflow.python import keras
from lib.tfcustom import gelu
from lib.math import f1_m


models = (
    "20200201-1609_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=0.1_zdim=8_anneal=1",
    "20200201-1925_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=0.1_zdim=8_anneal=5",
    "20200201-2318_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=0.1_zdim=8_anneal=20",
    "20200202-1036_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=1_zdim=8_anneal=1",
    "20200202-1724_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=1_zdim=8_anneal=5",
    "20200202-2353_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=1_zdim=8_anneal=20",
)

for model_name in models:
    latest_model_path = _latest_model("../models/{}".format(model_name))
    autoencoder = keras.models.load_model(
        latest_model_path,
        custom_objects = {"gelu": gelu, "f1_m": f1_m},
    )
    encoder = _get_encoding_layer(autoencoder)