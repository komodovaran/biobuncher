import tensorflow.keras.models as m

model = m.load_model("models/20200202-1036_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=1_zdim=8_anneal=1/model_018.h5")

print(model.summary())
# for l in model.layers:
#     print(l)