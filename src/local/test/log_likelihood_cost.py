import sys
sys.path.insert(0, '../../')
import theano
import theano.tensor as T
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer


def logLikelihoodCost(softmaxLayer, net):
    "Return the log-likelihood cost."
    return -T.mean(T.log(softmaxLayer.output_dropout)[T.arange(net.data.y.shape[0]), net.data.y])

def calcCost(cost, net):
    training_x, training_y = net.data.training_data
    fn = theano.function(
        [], cost, givens={
            net.data.x:
                training_x[0: net.mini_batch_size],
            net.data.y:
                training_y[0: net.mini_batch_size]
        })
    return fn()

mini_batch_size = 10
net = Network([
    FullyConnectedLayer(n_in=784, n_out=10),
    SoftmaxLayer(n_in=10, n_out=10)], mini_batch_size)
rightCost = calcCost(logLikelihoodCost(net.layers[-1], net), net)
print(rightCost)

newCost = calcCost(net.layers[-1].cost(net), net)
print(newCost)


