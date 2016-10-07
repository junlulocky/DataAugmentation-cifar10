import os

import numpy as np
import theano as tn

import sdeepy.utils.pylab as pl
from sdeepy.core.network import Sequential
from sdeepy.data_provider import DataProviderFromMemory
from sdeepy.edge.convolution import Convolution, MaxPooling, AveragePooling
from sdeepy.edge.activation import Relu, Tanh
from sdeepy.edge.affine import Affine
from sdeepy.edge.loss_function import CategoricalCrossentropy, ZeroOne
from sdeepy.edge.unclassified import Softmax
from sdeepy.monitor import MonitorConvolution, Monitor
from dataset_util import load_cifar10_validation_set
from sdeepy.support import solver
from sdeepy.optimize import GradientDescent
from sdeepy.core.save import save_net
from sdeepy.core.load import load_net
from sdeepy.optimize.penalty import L2

from morph_test.utils_local import *



net_file_name = os.path.dirname(__file__) + '/cifar10_pena.sdn'#.format(os.path.basename(__file__).replace('.py', ''))

save_path_std = os.path.dirname(__file__) + '/test'

PARAM = dict()
max_epoch = 90
batch_size = 500
hyp_param1 = 0.001

# Create data provider
trainX, trainY, num_train_samples, validX, validY, num_test_samples = load_cifar10_validation_set(rasterized=False)
trainDP = DataProviderFromMemory([trainX, trainY], batch_size, shuffle=True)
validDP = DataProviderFromMemory([validX, validY], batch_size)

print "change..."

#Build inital net
##################################
if os.path.isfile(net_file_name):
    print("Load inital net ...")
    net = load_net(net_file_name)
    C1 = net.get_edge_linear_order()[0]
    P1 = net.get_edge_linear_order()[1]
    C2 = net.get_edge_linear_order()[3]
    P2 = net.get_edge_linear_order()[4]
    C3 = net.get_edge_linear_order()[6]
    P3 = net.get_edge_linear_order()[7]
    A1 = net.get_edge_linear_order()[9]
    A2 = net.get_edge_linear_order()[11]
else:
    # Network configuration
    rng = np.random.RandomState()
    edges = [
        Convolution(
            inshape=(3, 32, 32), outmaps=32, kernel_shape=(5, 5,),
            with_bias=True, init_method='bengio2010_tanh',
            batch_size=batch_size, rng=rng, border_mode='same', strides=(1,)*2),
        MaxPooling(
            inshape=(32, 32, 32), pool_shape=(3, 3), strides=(2,)*2),
        Relu(),
        Convolution(
            inshape=(32, 15, 15), outmaps=32, kernel_shape=(5, 5,),
            with_bias=True, init_method='bengio2010_tanh',
            batch_size=batch_size, rng=rng, border_mode='same', strides=(1,)*2),
        AveragePooling(
            inshape=(32, 15, 15), pool_shape=(3, 3), strides=(2,)*2),
        Relu(),
        Convolution(
            inshape=(32, 7, 7), outmaps=64, kernel_shape=(5, 5,),
            with_bias=True, init_method='bengio2010_tanh',
            batch_size=batch_size, rng=rng, border_mode='same', strides=(1,) * 2),
        AveragePooling(
            inshape=(64, 7, 7), pool_shape=(3, 3), strides=(2,) * 2),
        Relu(),

        Affine(inshape=(64, 3, 3), outshape=(64,),
               with_bias=True, init_method='bengio2010_tanh', rng=rng,),
        Relu(),
        Affine(inshape=(64,), outshape=(10,),
               with_bias=True, init_method='bengio2010_tanh', rng=rng),
        Softmax()
    ]

    # Create convolutional neural net
    net = Sequential(edges, name='cnn')


    # Optimization for training
    opt = GradientDescent(net, losses=CategoricalCrossentropy(),
                          data_provider=trainDP,
                          param_penalties={edges[0].s_params['w']: L2(hyp_param1),
                                           edges[0].s_params['b']: L2(hyp_param1),
                                           edges[3].s_params['w']: L2(hyp_param1),
                                           edges[3].s_params['b']: L2(hyp_param1),
                                           edges[6].s_params['w']: L2(hyp_param1),
                                           edges[6].s_params['b']: L2(hyp_param1),
                                           edges[9].s_params['w']: L2(hyp_param1),
                                           edges[9].s_params['b']: L2(hyp_param1),
                                           edges[11].s_params['w']: L2(hyp_param1),
                                           edges[11].s_params['b']: L2(hyp_param1),
                                           },
                          updater=GradientDescent.Updater(
                                method='default', base_lrate=0.1)
                          )

    print('start training!')
    mon = [Monitor(net,
                   ZeroOne(),
                   trainDP,
                   validDP,
                   monitor_condition=lambda epoch: True,
                   name='Error',
                   save_path=save_path_std),
           Monitor(net,
                   CategoricalCrossentropy(),
                   trainDP,
                   validDP,
                   monitor_condition=lambda epoch: True,
                   name='Cost',
                   save_path=save_path_std)
           ]
    solver.train(opt, monitors=mon, max_epoch=max_epoch)
    save_net(net, net_file_name)


## evaluate net
orig_train_err, orig_train_cost = eval(net.forward([trainX])[0],trainY)
orig_test_err, orig_test_cost = eval(net.forward([validX])[0],validY)

######## Display results ##############
print "            || TRAIN COST  |  TRAIN ERROR | TEST ERROR  ||"
print "=========================================================="
print("Original     || {:.5f}     | {:.5f}     | {:.5f}     ||".format(orig_train_cost,orig_train_err*100,orig_test_err*100))





