# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import theano

import sdeepy.utils.pylab as pl
from sdeepy.core.network import Sequential
from sdeepy.data_provider import DataProviderFromMemory
from sdeepy.edge.convolution import Convolution, MaxPooling
from sdeepy.edge.activation import Relu
from sdeepy.edge import Dropout
from sdeepy.edge.affine import Affine
from sdeepy.edge.loss_function import CategoricalCrossentropy, ZeroOne
from sdeepy.edge.unclassified import Softmax
from sdeepy.monitor import Monitor
from sdeepy.support import solver
from sdeepy.optimize import AdamUpdater, GradientDescent
from sdeepy.core import save_net
from scipy.io import loadmat
from morph_test.cifar10.dataset_util import load_cifar10_validation_set
from sdeepy.optimize.penalty import L2

if __name__ == '__main__':
    print 'chang33333333..'
    save_path = os.path.dirname(__file__) + '/cifar10_allconv'
    batch_size = 128
    max_epoch = 200
    base_lrate = 1e-4
    hyp_param1 = 0.001

    # Create data provider
    train_x, train_y, num_train_samples, test_x, test_y, num_test_samples = load_cifar10_validation_set(rasterized=False)
    path_gcn_whitened = os.path.dirname(__file__) + '/tmp.cifar10_gcn_whitened.npz'

    if not os.path.isfile(path_gcn_whitened):
        raise RuntimeError()
    obj = np.load(path_gcn_whitened)
    floatX = theano.config.floatX
    train_x = obj['train'].astype(floatX)
    test_x = obj['valid'].astype(floatX)

    isFlipped = False
    if isFlipped:
        batch_size *= 2
        train_x_flipped = train_x[:,:,:,::-1]
        train_y_flipped = train_y

        train_x = np.concatenate((train_x, train_x_flipped), axis=0)
        train_y = np.concatenate((train_y, train_y_flipped), axis=0)
        print train_x.shape

    isCropped = False
    if isCropped:
        batch_size*=5
        train_x = np.pad(train_x, ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
        train_x = np.concatenate((train_x[:,:,0:32,0:32], train_x[:,:,8:40,0:32],train_x[:,:,0:32,8:40],train_x[:,:,8:40,8:40],train_x[:,:,4:36,4:36]), axis=0)
        train_y = np.concatenate((train_y, train_y,train_y,train_y,train_y), axis=0)
        print train_x.shape

    train_dp = DataProviderFromMemory([train_x, train_y], batch_size=batch_size, shuffle=True, modal_names=['x', 'y'])
    test_dp = DataProviderFromMemory([test_x, test_y], batch_size=batch_size, shuffle=True, modal_names=['x', 'y'])




    # Network configuration
    rng = np.random.RandomState()
    a = Affine(inshape=(10, 7, 7), outshape=(10,), with_bias=True, init_method='bengio2010_tanh', rng=rng),




    edges = [
        # 64C3-64C3-P2
        Convolution(inshape=(3, 32, 32), outmaps=96, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(96, 32, 32), outmaps=96, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(96, 32, 32), outmaps=96, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        MaxPooling(inshape=(96, 32, 32), pool_shape=(3, 3), strides=(2,) * 2),

        # 128C3-128C3-P2
        Convolution(inshape=(96, 15, 15), outmaps=192, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(192, 15, 15), outmaps=192, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(192, 15, 15), outmaps=192, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        MaxPooling(inshape=(192, 15, 15), pool_shape=(3, 3), strides=(2,) * 2),

        # 256C3-256C3-P2
        Convolution(inshape=(192, 7, 7), outmaps=192, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(192, 7, 7), outmaps=192, kernel_shape=(1, 1,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(192, 7, 7), outmaps=10, kernel_shape=(1, 1,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),

        Affine(inshape=(10, 7, 7), outshape=(10,), with_bias=True, init_method='bengio2010_tanh', rng=rng),
        Softmax()
    ]



    # Create convolutional neural net
    net = Sequential(edges, name='cnn')
    # Monitors
    monitors = [
        Monitor(
            net, ZeroOne(), train_dp, test_dp, name='Error',
            popup_figure=False, save_path=save_path, monitor_condition=lambda epoch: True
        ),
    ]

    # Optimization for training
    opt = GradientDescent(net, losses=CategoricalCrossentropy(),
                          data_provider=train_dp,
                          # param_penalties={edges[0].s_params['w']: L2(hyp_param1),
                          #                  edges[0].s_params['b']: L2(hyp_param1),
                          #                  edges[2].s_params['w']: L2(hyp_param1),
                          #                  edges[2].s_params['b']: L2(hyp_param1),
                          #                  edges[6].s_params['w']: L2(hyp_param1),
                          #                  edges[6].s_params['b']: L2(hyp_param1),
                          #                  edges[8].s_params['w']: L2(hyp_param1),
                          #                  edges[8].s_params['b']: L2(hyp_param1),
                          #                  edges[12].s_params['w']: L2(hyp_param1),
                          #                  edges[12].s_params['b']: L2(hyp_param1),
                          #                  edges[14].s_params['w']: L2(hyp_param1),
                          #                  edges[14].s_params['b']: L2(hyp_param1),
                          #                  edges[16].s_params['w']: L2(hyp_param1),
                          #                  edges[16].s_params['b']: L2(hyp_param1),
                          #                  edges[18].s_params['w']: L2(hyp_param1),
                          #                  edges[18].s_params['b']: L2(hyp_param1),
                          #                  edges[22].s_params['w']: L2(hyp_param1),
                          #                  edges[22].s_params['b']: L2(hyp_param1),
                          #                  edges[25].s_params['w']: L2(hyp_param1),
                          #                  edges[25].s_params['b']: L2(hyp_param1),
                          #                  edges[28].s_params['w']: L2(hyp_param1),
                          #                  edges[28].s_params['b']: L2(hyp_param1),
                          #                  },
                          updater=AdamUpdater(alpha=base_lrate))

    print('start training!')
    solver.train(opt, monitors=monitors, max_epoch=max_epoch)
    save_net(net, 'cifar10_allconv.sdn')
