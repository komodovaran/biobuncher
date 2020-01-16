import os

from tensorflow.python import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def main(path):
    model = keras.models.load_model(path) # type: keras.models.Model
    print("Activations:")
    for layer in model.layers:
        try:
            print(layer.activation)
        except AttributeError:
            pass
    print(model.get_layer("encoded").output.shape)

if __name__ == "__main__":
    PATH = "models/20191126-2337_multi_lstm_autoencoder_dim=32_variable_data=tracks-cme_split-c1_var.npz/model_067.h5"
    main(PATH)