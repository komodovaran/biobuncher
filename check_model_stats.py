from tensorflow.python import keras
import lib.models
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Models trained between 26-27th Nov
path = "models/20191126-2337_multi_lstm_autoencoder_dim=32_variable_data=tracks-cme_split-c1_var.npz/model_067.h5"

model = keras.models.load_model(path) # type: keras.models.Model

print("Activations:")
for layer in model.layers:
    try:
        print(layer.activation)
    except AttributeError:
        pass

print(model.get_layer("encoded").output.shape)