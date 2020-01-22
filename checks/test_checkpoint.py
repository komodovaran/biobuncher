import numpy as np
from numpy.testing import assert_allclose
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint

vec_size = 100
n_units = 10

x_train = np.random.rand(500, 10, vec_size)
y_train = np.random.rand(500, vec_size)

model = Sequential()
model.add(LSTM(n_units, input_shape=(None, vec_size), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_units, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_units))
model.add(Dropout(0.2))
model.add(Dense(vec_size, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# define the checkpoint
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(x_train, y_train, epochs=5, batch_size=50, callbacks=callbacks_list)

# load the model
new_model = load_model("model.h5")
assert_allclose(model.predict(x_train),
                new_model.predict(x_train),
                1e-5)

# fit the model
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
new_model.fit(x_train, y_train, epochs=5, batch_size=50, callbacks=callbacks_list)