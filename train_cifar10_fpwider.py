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
from sdeepy.edge.unclassified import BatchNormalization
from scipy.io import savemat, loadmat

if __name__ == '__main__':
    print 'chang123123..'
    save_path = os.path.dirname(__file__) + '/result_fpwider/scratch_batch500'
    batch_size = 500
    # max_epoch = 200
    #base_lrate = 1e-4
    hyp_param1 = 0.001



    # Create data provider
    '''BEGIN...'''
    train_x, train_y, num_train_samples, test_x, test_y, num_test_samples = load_cifar10_validation_set(rasterized=False)
    path_gcn_whitened = os.path.dirname(__file__) + '/tmp.cifar10_gcn_whitened.npz'

    if not os.path.isfile(path_gcn_whitened):
        raise RuntimeError()
    obj = np.load(path_gcn_whitened)
    floatX = theano.config.floatX
    train_x = obj['train'].astype(floatX)
    test_x = obj['valid'].astype(floatX)

    # whether flipped or not
    isFlipped = True
    if isFlipped:
        batch_size *= 2
        train_x_flipped = train_x[:,:,:,::-1]
        train_y_flipped = train_y

        train_x = np.concatenate((train_x, train_x_flipped), axis=0)
        train_y = np.concatenate((train_y, train_y_flipped), axis=0)
        print train_x.shape

    # whether cropped or not
    # isCropped = False
    # if isCropped:
    #     batch_size*=5
    #     train_x = np.pad(train_x, ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
    #     train_x = np.concatenate((train_x[:,:,0:32,0:32], train_x[:,:,8:40,0:32],train_x[:,:,0:32,8:40],train_x[:,:,8:40,8:40],train_x[:,:,4:36,4:36]), axis=0)
    #     train_y = np.concatenate((train_y, train_y,train_y,train_y,train_y), axis=0)
    #     print train_x.shape

    train_dp = DataProviderFromMemory([train_x, train_y], batch_size=batch_size, shuffle=True, modal_names=['x', 'y'])
    test_dp = DataProviderFromMemory([test_x, test_y], batch_size=batch_size, shuffle=True, modal_names=['x', 'y'])
    '''END...'''




    # Network configuration
    rng = np.random.RandomState()
    edges = [
        # 128C3-128C3-P2
        Convolution(inshape=(3, 32, 32), outmaps=128, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(128, 32, 32), outmaps=128, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        MaxPooling(inshape=(128, 32, 32), pool_shape=(2, 2)),
        Relu(),
        Dropout((128, 16, 16), p_default=0.25),

        # 256C3-256C3-P2
        Convolution(inshape=(128, 16, 16), outmaps=256, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(256, 16, 16), outmaps=256, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        MaxPooling(inshape=(256, 16, 16), pool_shape=(2, 2)),
        Relu(),
        Dropout((256, 8, 8), p_default=0.25),

        # 512C3-512C3-P2
        Convolution(inshape=(256, 8, 8), outmaps=512, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(512, 8, 8), outmaps=512, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        MaxPooling(inshape=(512, 8, 8), pool_shape=(2, 2)),
        Relu(),
        Dropout((512, 4, 4), p_default=0.25),

        # 1024FP-1024FP-10FP
        Affine(inshape=(512, 4, 4), outshape=(1024,), with_bias=True, init_method='bengio2010_tanh', rng=rng),
        Relu(),
        Dropout((1024,), p_default=0.5),
        Affine(inshape=(1024,), outshape=(1024,), with_bias=True, init_method='bengio2010_tanh', rng=rng),
        Relu(),
        Dropout((1024,), p_default=0.5),
        Affine(inshape=(1024,), outshape=(10,), with_bias=True, init_method='bengio2010_tanh', rng=rng),
        Softmax()
    ]


    # Create convolutional neural net
    net = Sequential(edges, name='cnn')
    # # Monitors
    # monitors = [
    #     Monitor(
    #         net, ZeroOne(), train_dp, test_dp, name='Error',
    #         popup_figure=False, save_path=save_path, monitor_condition=lambda epoch: True
    #     ),
    # ]
    #
    # # Optimization for training
    # opt = GradientDescent(net, losses=CategoricalCrossentropy(),
    #                       data_provider=train_dp,
    #                       updater=AdamUpdater(alpha=base_lrate))
    #
    # print('start training!')
    # solver.train(opt, monitors=monitors, max_epoch=max_epoch, save_path=save_path)
    # save_net(net, save_path+'/cifar10_fpwider.sdn')

    fine_tuning_epoch = 1000
    opt = GradientDescent(net,
                          losses=CategoricalCrossentropy(),
                          data_provider=train_dp,
                          updater=GradientDescent.Updater(method='default',
                                                          base_lrate=0.01, decay=1.0))
    mon = [Monitor(net,
                   ZeroOne(),
                   train_dp,
                   test_dp,
                   monitor_condition=lambda epoch: True,
                   save_path=save_path ,
                   name='Error'),
           Monitor(net,
                   CategoricalCrossentropy(),
                   train_dp,
                   test_dp,
                   monitor_condition=lambda epoch: True,
                   save_path=save_path,
                   name='Cost')
           ]

    costs = solver.train(opt, monitors=mon, max_epoch=fine_tuning_epoch, save_path=save_path )
    save_net(net, save_path+'/net_scratch'+str(fine_tuning_epoch)+'.sdn')

    train_errors = mon[0].errors_y_train
    valid_errors = mon[0].errors_y_valid
    train_costs = mon[1].errors_y_train

    RESULTS_FPWIDER_SCRATCH = dict()

    RESULTS_FPWIDER_SCRATCH['all_costs'] = list()
    RESULTS_FPWIDER_SCRATCH['all_train_errors'] = list()
    RESULTS_FPWIDER_SCRATCH['all_valid_errors'] = list()

    RESULTS_FPWIDER_SCRATCH['all_costs'] += list(train_costs)
    RESULTS_FPWIDER_SCRATCH['all_train_errors'] += list(train_errors)
    RESULTS_FPWIDER_SCRATCH['all_valid_errors'] += list(valid_errors)

    filename = save_path + '/RESULTS_FPWIDER_SCRATCH'+str(fine_tuning_epoch)+'.mat'
    dir = os.path.dirname(filename)

    try:
        os.stat(dir)
    except:
        os.mkdir(dir)

    savemat(filename, RESULTS_FPWIDER_SCRATCH)
    RESULTS_FPWIDER_SCRATCH = loadmat(filename, squeeze_me=True)
