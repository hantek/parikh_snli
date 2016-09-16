#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import os
import sys
import numpy
import cPickle
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.recurrent import Gate
from lasagne import init, nonlinearities

from util_layers import (DenseLayer3DInput, ComputeEmbeddingPool, AttendOnEmbedding, MeanOverDim)
from dataset import SNLI

import pdb
# from theano.compile.nanguardmode import NanGuardMode
theano.config.compute_test_value = 'off'  # 'off' # Use 'warn' to activate this feature


EMBDHIDA = int(sys.argv[1])         # 200
EMBDHIDB = int(sys.argv[2])         # 200
COMPHIDA = int(sys.argv[3])         # 200
COMPHIDB = int(sys.argv[4])         # 200
OUTHID = int(sys.argv[5])           # 200 Hidden unit numbers in output MLP
NROW = int(sys.argv[6])             # 20 Number of rows in matrix representation
DPOUT = float(sys.argv[7])          # 0.2 
LR = float(sys.argv[8])             # 0.01
WEDIM = int(sys.argv[9])            # 300 Dim of word embedding
BSIZE = int(sys.argv[10])           # 50 Minibatch size
GCLIP = int(sys.argv[11])           # 100 All gradients above this will be clipped
NEPOCH = int(sys.argv[12])          # 12 Number of epochs to train the net
STD = float(sys.argv[13])           # 0.1 Standard deviation of weights in initialization
UPDATEWE = bool(int(sys.argv[14]))  # 1 0 for False and 1 for True. Update word embedding in training
filename = __file__.split('.')[0] + \
           '_EMBDHIDA' + str(EMBDHIDA) + \
           '_EMBDHIDB' + str(EMBDHIDB) + \
           '_COMPHIDA' + str(COMPHIDA) + \
           '_COMPHIDB' + str(COMPHIDB) + \
           '_OUTHID' + str(OUTHID) + \
           '_NROW' + str(NROW) + \
           '_DPOUT' + str(DPOUT) + \
           '_LR' + str(LR) + \
           '_WEDIM' + str(WEDIM) + \
           '_BSIZE' + str(BSIZE) + \
           '_GCLIP' + str(GCLIP) + \
           '_NEPOCH' + str(NEPOCH) + \
           '_STD' + str(STD) + \
           '_UPDATEWE' + str(UPDATEWE)


def main(num_epochs=NEPOCH):
    print("Loading data ...")
    snli = SNLI(batch_size=BSIZE)
    train_batches = list(snli.train_minibatch_generator())
    dev_batches = list(snli.dev_minibatch_generator())
    test_batches = list(snli.test_minibatch_generator())
    W_word_embedding = snli.weight  # W shape: (# vocab size, WE_DIM)
    del snli

    print("Building network ...")
    ########### input layers ###########
    # hypothesis
    input_var_h = T.TensorType('int32', [False, False])('hypothesis_vector')
    input_var_h.tag.test_value = numpy.hstack((numpy.random.randint(1, 10000, (BSIZE, 18), 'int32'),
                                               numpy.zeros((BSIZE, 6)).astype('int32')))
    l_in_h = lasagne.layers.InputLayer(shape=(BSIZE, None), input_var=input_var_h)
    
    input_mask_h = T.TensorType('int32', [False, False])('hypo_mask')
    input_mask_h.tag.test_value = numpy.hstack((numpy.ones((BSIZE, 18), dtype='int32'),
                                                numpy.zeros((BSIZE, 6), dtype='int32')))
    input_mask_h.tag.test_value[1, 18:22] = 1
    l_mask_h = lasagne.layers.InputLayer(shape=(BSIZE, None), input_var=input_mask_h)
    
    # premise
    input_var_p = T.TensorType('int32', [False, False])('premise_vector')
    input_var_p.tag.test_value = numpy.hstack((numpy.random.randint(1, 10000, (BSIZE, 16), 'int32'),
                                               numpy.zeros((BSIZE, 3)).astype('int32')))
    l_in_p = lasagne.layers.InputLayer(shape=(BSIZE, None), input_var=input_var_p)
    
    input_mask_p = T.TensorType('int32', [False, False])('premise_mask')
    input_mask_p.tag.test_value = numpy.hstack((numpy.ones((BSIZE, 16), dtype='int32'),
                                                numpy.zeros((BSIZE, 3), dtype='int32')))
    input_mask_p.tag.test_value[1, 16:18] = 1
    l_mask_p = lasagne.layers.InputLayer(shape=(BSIZE, None), input_var=input_mask_p)
    ###################################

    # output shape (BSIZE, None, WEDIM)
    l_hypo_embed = lasagne.layers.EmbeddingLayer(
        l_in_h,
        input_size=W_word_embedding.shape[0],
        output_size=W_word_embedding.shape[1],
        W=W_word_embedding)
    
    l_prem_embed = lasagne.layers.EmbeddingLayer(
        l_in_p,
        input_size=W_word_embedding.shape[0],
        output_size=W_word_embedding.shape[1],
        W=l_hypo_embed.W)

    # ATTEND
    l_hypo_embed_dpout = lasagne.layers.DropoutLayer(l_hypo_embed, p=DPOUT, rescale=True)
    l_hypo_embed_hid1 = DenseLayer3DInput(
        l_hypo_embed_dpout, num_units=EMBDHIDA, nonlinearity=lasagne.nonlinearities.rectify)
    l_hypo_embed_hid1_dpout = lasagne.layers.DropoutLayer(l_hypo_embed_hid1, p=DPOUT, rescale=True)
    l_hypo_embed_hid2 = DenseLayer3DInput(
        l_hypo_embed_hid1_dpout, num_units=EMBDHIDB, nonlinearity=lasagne.nonlinearities.rectify)

    l_prem_embed_dpout = lasagne.layers.DropoutLayer(l_prem_embed, p=DPOUT, rescale=True)
    l_prem_embed_hid1 = DenseLayer3DInput(
        l_prem_embed_dpout, num_units=EMBDHIDA, nonlinearity=lasagne.nonlinearities.rectify)
    l_prem_embed_hid1_dpout = lasagne.layers.DropoutLayer(l_prem_embed_hid1, p=DPOUT, rescale=True)
    l_prem_embed_hid2 = DenseLayer3DInput(
        l_prem_embed_hid1_dpout, num_units=EMBDHIDB, nonlinearity=lasagne.nonlinearities.rectify)
    
    # output dim: (BSIZE, NROWx, NROWy)
    l_e = ComputeEmbeddingPool([l_hypo_embed_hid1, l_prem_embed_hid2])
    # output dim: (BSIZE, NROWy, DIM)
    l_hypo_weighted = AttendOnEmbedding([l_hypo_embed, l_e], masks=[l_mask_h, l_mask_p], direction='col')
    # output dim: (BSIZE, NROWx, DIM)
    l_prem_weighted = AttendOnEmbedding([l_prem_embed, l_e], masks=[l_mask_h, l_mask_p], direction='row')

    # COMPARE
    # output dim: (BSIZE, NROW, 4*LSTMHID)
    l_hypo_premwtd = lasagne.layers.ConcatLayer([l_hypo_embed, l_prem_weighted], axis=2)
    l_prem_hypowtd = lasagne.layers.ConcatLayer([l_prem_embed, l_hypo_weighted], axis=2)

    l_hypo_premwtd_dpout = lasagne.layers.DropoutLayer(l_hypo_premwtd, p=DPOUT, rescale=True)
    l_hypo_comphid1 = DenseLayer3DInput(
        l_hypo_premwtd_dpout, num_units=COMPHIDA, nonlinearity=lasagne.nonlinearities.rectify)
    
    l_hypo_comphid1_dpout = lasagne.layers.DropoutLayer(l_hypo_comphid1, p=DPOUT, rescale=True)
    l_hypo_comphid2 = DenseLayer3DInput(
        l_hypo_comphid1_dpout, num_units=COMPHIDB, nonlinearity=lasagne.nonlinearities.rectify)

    l_prem_hypowtd_dpout = lasagne.layers.DropoutLayer(l_prem_hypowtd, p=DPOUT, rescale=True)
    l_prem_comphid1 = DenseLayer3DInput(
        l_prem_hypowtd_dpout, num_units=COMPHIDA,
        W=l_hypo_comphid1.W, b=l_hypo_comphid1.b, nonlinearity=lasagne.nonlinearities.rectify)
    l_prem_comphid1_dpout = lasagne.layers.DropoutLayer(l_prem_comphid1, p=DPOUT, rescale=True)
    l_prem_comphid2 = DenseLayer3DInput(
        l_prem_comphid1_dpout, num_units=COMPHIDB,
        W=l_hypo_comphid2.W, b=l_hypo_comphid2.b, nonlinearity=lasagne.nonlinearities.rectify)

    # AGGREGATE
    # output dim: (BSIZE, 4*LSTMHID)
    l_hypo_mean = MeanOverDim(l_hypo_comphid2, mask=l_mask_h, dim=1)
    l_prem_mean = MeanOverDim(l_prem_comphid2, mask=l_mask_p, dim=1)

    l_v1v2 = lasagne.layers.ConcatLayer([l_hypo_mean, l_prem_mean], axis=1)

    l_v1v2_dpout = lasagne.layers.DropoutLayer(l_v1v2, p=DPOUT, rescale=True)
    l_outhid = lasagne.layers.DenseLayer(
        l_v1v2_dpout, num_units=OUTHID, nonlinearity=lasagne.nonlinearities.rectify)

    l_outhid_dpout = lasagne.layers.DropoutLayer(l_outhid, p=DPOUT, rescale=True)
    l_output = lasagne.layers.DenseLayer(
        l_outhid_dpout, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)


    ########### target, cost, validation, etc. ##########
    target_values = T.ivector('target_output')
    target_values.tag.test_value = numpy.asarray([1,] * BSIZE, dtype='int32')

    network_output = lasagne.layers.get_output(l_output)
    network_prediction = T.argmax(network_output, axis=1)
    error_rate = T.mean(T.neq(network_prediction, target_values))
    
    network_output_clean = lasagne.layers.get_output(l_output, deterministic=True) 
    network_prediction_clean = T.argmax(network_output_clean, axis=1) 
    error_rate_clean = T.mean(T.neq(network_prediction_clean, target_values)) 

    cost = T.mean(T.nnet.categorical_crossentropy(network_output, target_values))
    cost_clean = T.mean(T.nnet.categorical_crossentropy(network_output_clean, target_values))

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_output)
    if not UPDATEWE:
        all_params.remove(l_hypo_embed.W)

    numparams = sum([numpy.prod(i) for i in [i.shape.eval() for i in all_params]])
    print("Number of params: {}\nName\t\t\tShape\t\t\tSize".format(numparams))
    print("-----------------------------------------------------------------")
    for item in all_params:
        print("{0:24}{1:24}{2}".format(item, item.shape.eval(), numpy.prod(item.shape.eval())))

    # if exist param file then load params
    look_for = 'params' + os.sep + 'params_' + filename + '.pkl'
    if os.path.isfile(look_for):
        print("Resuming from file: " + look_for)
        all_param_values = cPickle.load(open(look_for, 'rb'))
        for p, v in zip(all_params, all_param_values):
            p.set_value(v)
   
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LR)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function(
        [l_in_h.input_var, l_mask_h.input_var,
         l_in_p.input_var, l_mask_p.input_var, target_values],
        [cost, error_rate], updates=updates)
        # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))
    compute_cost = theano.function(
        [l_in_h.input_var, l_mask_h.input_var,
         l_in_p.input_var, l_mask_p.input_var, target_values],
        [cost_clean, error_rate_clean])
        # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))

    def evaluate(mode):
        if mode == 'dev':
            data = dev_batches
        if mode == 'test':
            data = test_batches
        
        set_cost = 0.
        set_error_rate = 0.
        for batches_seen, (hypo, hm, premise, pm, truth) in enumerate(data, 1):
            _cost, _error = compute_cost(hypo, hm, premise, pm, truth)
            set_cost = (1.0 - 1.0 / batches_seen) * set_cost + \
                       1.0 / batches_seen * _cost
            set_error_rate = (1.0 - 1.0 / batches_seen) * set_error_rate + \
                             1.0 / batches_seen * _error
        
        return set_cost, set_error_rate
    
    print("Done. Evaluating scratch model ...")
    dev_set_cost,  dev_set_error  = evaluate('dev')
    print("BEFORE TRAINING: dev cost %f, error %f" % (dev_set_cost,  dev_set_error))
    print("Training ...")
    try:
        for epoch in range(num_epochs):
            train_set_cost = 0.
            train_set_error = 0.
            start = time.time()
            
            for batches_seen, (hypo, hm, premise, pm, truth) in enumerate(train_batches, 1):
                _cost, _error = train(hypo, hm, premise, pm, truth)
                train_set_cost = (1.0 - 1.0 / batches_seen) * train_set_cost + \
                                 1.0 / batches_seen * _cost
                train_set_error = (1.0 - 1.0 / batches_seen) * train_set_error + \
                                  1.0 / batches_seen * _error
                if batches_seen % 100 == 0:
                    end = time.time()
                    print("Sample %d %.2fs, lr %.4f, train cost %f, error %f"  % (
                        batches_seen * BSIZE,
                        end - start,
                        LR,
                        train_set_cost,
                        train_set_error))
                    start = end

                if batches_seen % 2000 == 0:
                    dev_set_cost,  dev_set_error  = evaluate('dev')
                    print("***dev cost %f, error %f" % (dev_set_cost,  dev_set_error))

            # save parameters
            all_param_values = [p.get_value() for p in all_params]
            cPickle.dump(all_param_values,
                         open('params' + os.sep + 'params_' + filename + '.pkl', 'wb'))

            dev_set_cost,  dev_set_error  = evaluate('dev')
            test_set_cost, test_set_error = evaluate('test')

            print("epoch %d, cost: train %f dev %f test %f;\n"
                  "         error train %f dev %f test %f" % (
                epoch,
                train_set_cost,     dev_set_cost,   test_set_cost,
                train_set_error,    dev_set_error,  test_set_error))
    except KeyboardInterrupt:
        pdb.set_trace()
        pass

if __name__ == '__main__':
    main()

