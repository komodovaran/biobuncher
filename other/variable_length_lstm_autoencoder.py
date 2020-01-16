import os

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import CuDNNLSTM, Input, Lambda
from tensorflow.python.keras.models import Model
import lib.utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

class SingleBatchGenerator:
    def __init__(self, X):
        self.X = X

    def __call__(self):
        for i in range(len(self.X)):
            xi = np.expand_dims(self.X[i], axis=0)
            yield xi, xi

def variable_lstm_autoencoder(latent_dim, n_features):
    """
    Variable length autoencoder (works only with batch size 1)

    To get the encoder, make a new model with the encoding LSTM layer:
    encoder = Model(inputs = autoencoder.input,
                    outputs = autoencoder.get_layer("encoded").output)
    """

    def variable_repeat(x):
        # matrix with ones, shaped as (batch, steps, 1)
        step_matrix = K.ones_like(x[0][:, :, :1])
        # latent vars, shaped as (batch, 1, latent_dim)
        latent_matrix = K.expand_dims(x[1], axis=1)
        return K.batch_dot(step_matrix, latent_matrix)

    n_timesteps = None

    inputs = Input(shape=(n_timesteps, n_features))
    encoded = CuDNNLSTM(latent_dim, name="encoder")(inputs)
    decoded = Lambda(variable_repeat)([inputs, encoded])
    outputs = CuDNNLSTM(n_features, return_sequences=True, name = "decoder")(decoded)
    autoencoder = Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

X = [np.ones((np.random.randint(10, 100), 2)) for _ in range(12)]
gen = SingleBatchGenerator(X)
model = variable_lstm_autoencoder(latent_dim=10, n_features=2)

ds = tf.data.Dataset.from_generator(
    generator = gen,
    output_types=(tf.float64, tf.float64),
    output_shapes=((1, None, 2), (1, None, 2)),
)

model.fit(ds.repeat(), steps_per_epoch=len(X), epochs=100)

encoder = Model(inputs = model.input,
                outputs = model.get_layer("encoder").output)

features = encoder.predict_generator(ds, steps = len(X))
X_pred = [model.predict_on_batch(np.expand_dims(xi, axis = 0)) for xi in X]