import numpy
import theano
import theano.tensor as T

from lasagne import nonlinearities, init
from lasagne.layers.base import Layer, MergeLayer

import pdb


class DenseLayer3DInput(Layer):
    """
    Apply a 2D matrix to a 3D input, so its a batched dot with shared slices.
    
    Input: (BATCH_SIZE, inputdim1, inputdim2)

    Weight: 
    Depending on whether the weight is multiplied from left side of input,
    there are two shapes:
        right multiply case: (inputdim2, num_units)

    Output:
    
    Params:
        incoming,
        units_per_row,
        W
        b
        leftmul : True if the weight is left multiplied to the input.
        nonlinearity
        **kwargs
    """
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.tanh,
                 **kwargs):
        super(DenseLayer3DInput, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = self.input_shape[2]

        self.W = self.add_param(W, (num_inputs, num_units), name="W2D")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b2D",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, **kwargs):
        
        # pdb.set_trace()

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 'x', 0)
        return self.nonlinearity(activation)


class ComputeEmbeddingPool(MergeLayer):
    """
    Input :
        x: (BSIZE, NROW, DIM)
        y: (BSIZE, NROW, DIM)
    Output :
        (BSIZE, NROW, NROW)
    """
    def __init__(self, incomings, **kwargs):
        super(ComputeEmbeddingPool, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        xshape = input_shapes[0]
        yshape = input_shapes[1]
        return (xshape[0], xshape[1], yshape[1])

    def get_output_for(self, inputs, **kwargs):
        x = inputs[0]
        y = inputs[1]
        return T.batched_dot(x, y.dimshuffle(0, 2, 1))


class AttendOnEmbedding(MergeLayer):
    """
    incomings=[x, embeddingpool], masks=[xmask, ymask], direction='col'
    or
              [y, embeddingpool], masks=[xmask, ymask], direction='row'
    
    Output :
              alpha; or beta
    """
    def __init__(self, incomings, masks=None, direction='col', **kwargs):
        self.have_mask = False
        if masks:
            incomings = incomings + masks
            self.have_mask = True
        super(AttendOnEmbedding, self).__init__(incomings, **kwargs)
        self.direction = direction

    def get_output_shape_for(self, input_shapes):
        sent_shape = input_shapes[0]
        emat_shape = input_shapes[1]
        if self.direction == 'col':
            # x:    (BSIZE, R_x, DIM)
            # emat: (BSIZE. R_x, R_y)
            # out:  (BSIZE, R_y, DIM)
            return (sent_shape[0], emat_shape[2], sent_shape[2])
        elif self.direction == 'row':
            # y:    (BSIZE, R_y, DIM)
            # emat: (BSIZE. R_x, R_y)
            # out:  (BSIZE, R_x, DIM)
            return (sent_shape[0], emat_shape[1], sent_shape[2])

    def get_output_for(self, inputs, **kwargs):
        sentence = inputs[0]
        emat = inputs[1]
        if self.have_mask:
            xmask = inputs[2]
            ymask = inputs[3]
            xymask = T.batched_dot(xmask.dimshuffle(0, 1, 'x'),
                                   ymask.dimshuffle(0, 'x', 1))
            emat = emat * xymask.astype(theano.config.floatX) - \
                   numpy.asarray(1e36).astype(theano.config.floatX) * \
                   (1 - xymask).astype(theano.config.floatX)

        if self.direction == 'col':  # softmax on x's dim, and multiply by x
            annotation = T.nnet.softmax(
                emat.dimshuffle(0, 2, 1).reshape((
                    emat.shape[0] * emat.shape[2], emat.shape[1]))
            ).reshape((
                emat.shape[0], emat.shape[2], emat.shape[1]
            ))  # (BSIZE, R_y, R_x)
            if self.have_mask:
                annotation = annotation * ymask.dimshuffle(
                    0, 1, 'x').astype(theano.config.floatX)
        elif self.direction == 'row':  # softmax on y's dim, and multiply by y
            annotation = T.nnet.softmax(
                emat.reshape((
                    emat.shape[0] * emat.shape[1], emat.shape[2]))
            ).reshape((
                emat.shape[0], emat.shape[1], emat.shape[2]
            ))  # (BSIZE, R_x, R_y)
            if self.have_mask:
                annotation = annotation * xmask.dimshuffle(
                    0, 1, 'x').astype(theano.config.floatX)
        return T.batched_dot(annotation, sentence)


class MeanOverDim(MergeLayer):
    """
    dim can be a number or a tuple of numbers to indicate which dim to compute mean.
    """
    def __init__(self, incoming, mask=None, dim=1, **kwargs):
        incomings = [incoming]
        self.have_mask = False
        if mask:
            incomings.append(mask)
            self.have_mask = True
        super(MeanOverDim, self).__init__(incomings, **kwargs)
        self.dim = dim

    def get_output_shape_for(self, input_shapes):
        return tuple(x for i, x in enumerate(input_shapes[0]) if i != self.dim)

    def get_output_for(self, inputs, **kwargs):
        if self.have_mask:
            return T.sum(inputs[0], axis=self.dim) / \
                   inputs[1].sum(axis=1).dimshuffle(0, 'x')
        else:
            return T.mean(inputs[0], axis=self.dim)
