import os

CIFAR10_DATA_DIR = os.path.join(os.environ['HOME'], 'data/tensorflow/cifar10/cifar-10-batches-bin')
MNIST_DATA_DIR = os.path.join(os.environ['HOME'], 'data/tensorflow/MNIST_data')
RESNET_EPOCH_SIZE = 50000
ALEXNET_EPOCH_SIZE = 50000
FCN_EPOCH_SIZE = 60000


class Framework(object):
    tensorflow = 'tensorflow'


class FCN(object):
    fcn5 = 'fcn5'


class CNN(object):
    alexnet = 'alexnet'
    resnet = 'resnet'


class RNN(object):
    lstm = 'lstm'


class NetworkType(object):
    fc = 'fc'
    cnn = 'cnn'
    rnn = 'rnn'


class Synthetic(object):
    true = '1'
    false = '0'


class Status(object):
    enabled = '1'
    disabled = '0'

