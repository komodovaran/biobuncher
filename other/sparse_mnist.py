'''Keras implementation of the k-sparse autoencoder.
'''
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from lib.tfcustom import KSparse, UpdateSparsityLevel, calculate_sparsity_levels

if __name__ == "__main__":
    '''Example of how to use the k-sparse autoencoder to learn sparse features of MNIST digits.'''
    # Process MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, y_train = shuffle(x_train, y_train, random_state = 1)
    x_test, y_test = shuffle(x_test, y_test, random_state = 1)

    def process(x):
        return x.reshape(x.shape[0], x.shape[1] ** 2) / 255

    x_train = process(x_train)
    x_test = process(x_test)

    x_train, x_val = train_test_split(x_train, test_size = 10000, random_state = 1)

    # Define autoencoder parameters
    epochs = 30
    embedding_size = 128
    initial_sparsity = 128
    final_sparsity = 15
    sparsity_levels = calculate_sparsity_levels(initial_sparsity, final_sparsity, epochs)

    # Build the k-sparse autoencoder
    inputs = Input((x_train.shape[1],))
    h = Dense(embedding_size, activation = 'sigmoid')(inputs)
    k_sparse = KSparse(sparsity_levels = sparsity_levels, name = 'KSparse')(h)
    outputs = Dense(x_train.shape[1], activation = 'sigmoid')(k_sparse)

    ae1 = Model(inputs, outputs)
    ae1.compile('adamax', 'mse')
    ae1.fit(x_train, x_train, validation_data = (x_val, x_val), epochs = epochs,
            callbacks = [UpdateSparsityLevel(),
                         ModelCheckpoint(filepath = "test.h5", save_best_only = True, save_weights_only = True)])

    ae2 = Model(inputs, outputs)
    ae2.load_weights("test.h5")
    ae2.compile('adamax', 'mse')

    encoder = Model(inputs, ae2.get_layer("KSparse").output)
    # Calculate sparse encodings
    encoded = encoder.predict(x_test)
