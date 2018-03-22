import os

CIFAR10_DATA_DIR = os.path.join(os.environ['HOME'], 'data/tensorflow/cifar10/cifar-10-batches-bin')
MNIST_DATA_DIR = os.path.join(os.environ['HOME'], 'data/tensorflow/MNIST_data')


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


class Status(object):
    enabled = '1'
    disabled = '0'

