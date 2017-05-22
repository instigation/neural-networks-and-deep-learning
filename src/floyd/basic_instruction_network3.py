import sys
sys.path.insert(0, '/code/')

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer


mini_batch_size = 10
net = Network([
    FullyConnectedLayer(n_in=784, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size, "/input/mnist.pkl.gz")
net.SGD(30, mini_batch_size, 0.1)