import os
import cPickle
import theano
import numpy
import warnings

import pdb


class SNLI(object):
    def __init__(self, batch_size=50, loadall=False,
                 datapath="/home/hantek/datasets/SNLI_GloVe_converted"):
        self.batch_size = batch_size
        self.datapath = datapath
        
        data_file = open(self.datapath, 'rb')
        cPickle.load(data_file)
        cPickle.load(data_file)
        self.train_set, self.dev_set, self.test_set = cPickle.load(data_file)
        self.weight = cPickle.load(data_file).astype(theano.config.floatX)
        if loadall:
            self.word2embed = cPickle.load(data_file)   # key: word, value: embedding
            self.word2num = cPickle.load(data_file)     # key: word, value: number
            self.num2word = cPickle.load(data_file)     # key: number, value: word
        data_file.close()

        self.train_size = len(self.train_set)
        self.dev_size = len(self.dev_set)
        self.test_size = len(self.test_set)
        self.train_ptr = 0
        self.dev_ptr = 0
        self.test_ptr = 0

    def train_minibatch_generator(self):
        while self.train_ptr <= self.train_size - self.batch_size:
            self.train_ptr += self.batch_size
            minibatch = self.train_set[self.train_ptr - self.batch_size : self.train_ptr]
            if len (minibatch) < self.batch_size:
                warnings.warn("There will be empty slots in minibatch data.", UserWarning)
            
            longest_hypo, longest_premise = \
                numpy.max(map(lambda x: (len(x[0]), len(x[1])), minibatch), axis=0)

            hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            truth = numpy.zeros((self.batch_size,), dtype='int32')
            mask_hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            mask_premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            for i, (h, p, t) in enumerate(minibatch):
                hypos[i, :len(h)] = h
                mask_hypos[i, :len(h)] = (1,) * len(h)
                premises[i, :len(p)] = p
                mask_premises[i, :len(p)] = (1,) * len(p)
                truth[i] = t
            
            yield hypos, mask_hypos, premises, mask_premises, truth

        else:
            self.train_ptr = 0
            raise StopIteration

    def dev_minibatch_generator(self, ):
        while self.dev_ptr <= self.dev_size - self.batch_size:
            self.dev_ptr += self.batch_size
            minibatch = self.dev_set[self.dev_ptr - self.batch_size : self.dev_ptr]
            if len (minibatch) < self.batch_size:
                warnings.warn("There will be empty slots in minibatch data.", UserWarning)

            longest_hypo, longest_premise = \
                numpy.max(map(lambda x: (len(x[0]), len(x[1])), minibatch), axis=0)

            hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            truth = numpy.zeros((self.batch_size,), dtype='int32')
            mask_hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            mask_premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            for i, (h, p, t) in enumerate(minibatch):
                hypos[i, :len(h)] = h
                mask_hypos[i, :len(h)] = (1,) * len(h)
                premises[i, :len(p)] = p
                mask_premises[i, :len(p)] = (1,) * len(p)
                truth[i] = t

            yield hypos, mask_hypos, premises, mask_premises, truth

        else:
            self.dev_ptr = 0
            raise StopIteration

    def test_minibatch_generator(self, ):
        while self.test_ptr <= self.test_size - self.batch_size:
            self.test_ptr += self.batch_size
            minibatch = self.test_set[self.test_ptr - self.batch_size : self.test_ptr]
            if len (minibatch) < self.batch_size:
                warnings.warn("There will be empty slots in minibatch data.", UserWarning)

            longest_hypo, longest_premise = \
                numpy.max(map(lambda x: (len(x[0]), len(x[1])), minibatch), axis=0)

            hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            truth = numpy.zeros((self.batch_size,), dtype='int32')
            mask_hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            mask_premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            for i, (h, p, t) in enumerate(minibatch):
                hypos[i, :len(h)] = h
                mask_hypos[i, :len(h)] = (1,) * len(h)
                premises[i, :len(p)] = p
                mask_premises[i, :len(p)] = (1,) * len(p)
                truth[i] = t

            yield hypos, mask_hypos, premises, mask_premises, truth

        else:
            self.test_ptr = 0
            raise StopIteration


class SICK(object):
    def __init__(self, batch_size=50, loadall=False, augment=False,
                 datapath="/Users/johanlin/Datasets/SICK/"):
        #          datapath="/home/hantek/datasets/SNLI_GloVe_converted"):
        self.batch_size = batch_size
        if augment:
            self.datapath = datapath + os.sep + 'SICK_augmented.pkl'
        else:
            self.datapath = datapath + os.sep + 'SICK.pkl'
        
        data_file = open(self.datapath, 'rb')
        cPickle.load(data_file)
        cPickle.load(data_file)
        self.train_set, self.dev_set, self.test_set = cPickle.load(data_file)
        self.weight = cPickle.load(data_file).astype(theano.config.floatX)
        if loadall:
            self.word2embed = cPickle.load(data_file)   # key: word, value: embedding
            self.word2num = cPickle.load(data_file)     # key: word, value: number
            self.num2word = cPickle.load(data_file)     # key: number, value: word
        data_file.close()

        self.train_size = len(self.train_set)
        self.dev_size = len(self.dev_set)
        self.test_size = len(self.test_set)
        self.train_ptr = 0
        self.dev_ptr = 0
        self.test_ptr = 0

    def train_minibatch_generator(self):
        while self.train_ptr <= self.train_size - self.batch_size:
            self.train_ptr += self.batch_size
            minibatch = self.train_set[self.train_ptr - self.batch_size : self.train_ptr]
            if len (minibatch) < self.batch_size:
                warnings.warn("There will be empty slots in minibatch data.", UserWarning)
            
            longest_hypo, longest_premise = \
                numpy.max(map(lambda x: (len(x[0]), len(x[1])), minibatch), axis=0)

            hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            truth = numpy.zeros((self.batch_size,), dtype='int32')
            mask_hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            mask_premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            for i, (h, p, t) in enumerate(minibatch):
                hypos[i, :len(h)] = h
                mask_hypos[i, :len(h)] = (1,) * len(h)
                premises[i, :len(p)] = p
                mask_premises[i, :len(p)] = (1,) * len(p)
                truth[i] = t
            
            yield hypos, mask_hypos, premises, mask_premises, truth

        else:
            self.train_ptr = 0
            raise StopIteration

    def dev_minibatch_generator(self, ):
        while self.dev_ptr <= self.dev_size - self.batch_size:
            self.dev_ptr += self.batch_size
            minibatch = self.dev_set[self.dev_ptr - self.batch_size : self.dev_ptr]
            if len (minibatch) < self.batch_size:
                warnings.warn("There will be empty slots in minibatch data.", UserWarning)

            longest_hypo, longest_premise = \
                numpy.max(map(lambda x: (len(x[0]), len(x[1])), minibatch), axis=0)

            hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            truth = numpy.zeros((self.batch_size,), dtype='int32')
            mask_hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            mask_premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            for i, (h, p, t) in enumerate(minibatch):
                hypos[i, :len(h)] = h
                mask_hypos[i, :len(h)] = (1,) * len(h)
                premises[i, :len(p)] = p
                mask_premises[i, :len(p)] = (1,) * len(p)
                truth[i] = t

            yield hypos, mask_hypos, premises, mask_premises, truth

        else:
            self.dev_ptr = 0
            raise StopIteration

    def test_minibatch_generator(self, ):
        while self.test_ptr <= self.test_size - self.batch_size:
            self.test_ptr += self.batch_size
            minibatch = self.test_set[self.test_ptr - self.batch_size : self.test_ptr]
            if len (minibatch) < self.batch_size:
                warnings.warn("There will be empty slots in minibatch data.", UserWarning)

            longest_hypo, longest_premise = \
                numpy.max(map(lambda x: (len(x[0]), len(x[1])), minibatch), axis=0)

            hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            truth = numpy.zeros((self.batch_size,), dtype='int32')
            mask_hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            mask_premises = numpy.zeros((self.batch_size, longest_premise), dtype='int32')
            for i, (h, p, t) in enumerate(minibatch):
                hypos[i, :len(h)] = h
                mask_hypos[i, :len(h)] = (1,) * len(h)
                premises[i, :len(p)] = p
                mask_premises[i, :len(p)] = (1,) * len(p)
                truth[i] = t

            yield hypos, mask_hypos, premises, mask_premises, truth

        else:
            self.test_ptr = 0
            raise StopIteration

