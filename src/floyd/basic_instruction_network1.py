import sys
sys.path.insert(0, '/code/')

import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper("/input/mnist.pkl.gz")
net = network.Network([784, 10])
net.SGD(training_data, 10, 10, 3.0, test_data=test_data)