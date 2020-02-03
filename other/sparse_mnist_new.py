import ksparse.utils.mnist.mnist_helper as mh
from ksparse.layers.linear_layer import LinearLayer
from ksparse.layers.sparse_layer import SparseLayer
from ksparse.nets.fcnn import *
from ksparse.utils.activations import *
from ksparse.utils.cost_functions import *
from tensorflow.keras.layers import Dense

img_size = 28
num_hidden = 100
k = 70
learning_rate = 0.01
epochs = 10000
batch_size = 256
print_epochs = 1000
num_test_examples = 10

helper = mh.mnist_helper()
train_lbl, train_img, test_lbl, test_img = helper.get_data()

x_data = train_img.reshape(-1, img_size * img_size) / np.float32(256)
test_data = test_img.reshape(-1, img_size * img_size) / np.float32(256)

layers = [
    # LinearLayer(name="input", n_in=x_data.shape[1], n_out=num_hidden, activation=sigmoid_function),
    Dense(units = 32, activation = "relu", input_shape = x_data.shape[1]),
    SparseLayer(
        name="hidden 1",
        n_in=x_data.shape[1],
        n_out=num_hidden,
        activation=sigmoid_function,
        num_k_sparse=k,
    ),
    LinearLayer(
        name="output",
        n_in=num_hidden,
        n_out=x_data.shape[1],
        activation=sigmoid_function,
    ),
]

nn = FCNeuralNet(layers=layers, cost_func=subtract_err)
nn.print_network()

nn.train(
    x_data,
    x_data,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    print_epochs=print_epochs,
)
