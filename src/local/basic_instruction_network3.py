import sys
sys.path.insert(0, '../')

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
mini_batch_size = 10
net = Network([
    FullyConnectedLayer(n_in=784, n_out=10),
    SoftmaxLayer(n_in=10, n_out=10)], mini_batch_size)
net.SGD(1, mini_batch_size, 0.1)