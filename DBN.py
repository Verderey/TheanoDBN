"""
Author: 
Reuben Feinman
Brown University '15

Stencil code provided by www.deeplearning.com/tutorial/code
"""

import os
import sys
import time
from collections import OrderedDict

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer, DropoutHiddenLayer, _dropout_from_layer
from rbm import RBM, GBRBM, DropoutRBM


class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, hidden_layers_sizes, dropout_rates, 
                theano_rng=None, n_ins=784, n_outs=10, rmsprop=True):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.dropout = any([rate not in [1.0, 1] for rate in dropout_rates])
        self.rmsprop = rmsprop
        self.sigmoid_layers = []
        self.dropout_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        if dropout_rates:
            self.dropout = True
        else:
            self.dropout = False

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] labels
        # end-snippet-1
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
                next_dropout_layer_input = _dropout_from_layer(numpy_rng, layer_input, dropout_rates[0])
            else:
                layer_input = self.sigmoid_layers[-1].output
                next_dropout_layer_input = self.dropout_layers[-1].output

            next_dropout_layer = DropoutHiddenLayer(rng=numpy_rng,
                                        input=next_dropout_layer_input,
                                        activation=T.nnet.sigmoid,
                                        n_in=input_size, 
                                        n_out=hidden_layers_sizes[i],
                                        dropout_rate=dropout_rates[i+1])
            self.dropout_layers.append(next_dropout_layer)


            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid,
                                        W=next_dropout_layer.W * dropout_rates[i],
                                        b=next_dropout_layer.b)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # Construct an RBM that shared weights with this layer
            if i == 0:
                rbm_layer = RBM(numpy_rng=numpy_rng,
                                theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layers_sizes[i],
                                W=next_dropout_layer.W,
                                hbias=next_dropout_layer.b,
                                rmsprop=True)
            else:  
                rbm_layer = RBM(numpy_rng=numpy_rng,
                                theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layers_sizes[i],
                                W=next_dropout_layer.W,
                                hbias=next_dropout_layer.b,
                                rmsprop=True)

            self.rbm_layers.append(rbm_layer)

        # Output layer
        dropoutLogLayer = LogisticRegression(
            input=self.dropout_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.dropout_layers.append(dropoutLogLayer)

        # We now need to add a logistic layer on top of the MLP
        logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs,
            W=dropoutLogLayer.W * dropout_rates[-1],
            b=dropoutLogLayer.b)
        self.sigmoid_layers.append(logLayer)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors=self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.sigmoid_layers[-1].negative_log_likelihood
        self.errors = self.sigmoid_layers[-1].errors

        # Grab all the parameters for dropout_layers together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]


    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use
        momentum = T.scalar('momentum')

        # number of batches
        n_batches = train_set_x.get_value().shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                                 persistent=None, k=k, mom=momentum)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1), theano.Param(momentum, default=0.005)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, lr, mom):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''
        momentum = T.cast(mom, dtype=theano.config.floatX)

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value().shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value().shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        if self.dropout:
            train_cost = self.dropout_negative_log_likelihood(self.y)
        else:
            train_cost = self.negative_log_likelihood(self.y)
        errors = self.errors(self.y)

        def rmsprop(cost, learning_rate, r=0.9, epsilon=1e-2):
            rho = T.cast(r, dtype=theano.config.floatX)

            updates = []
            oldUpdates = []
            oldMeanSquares = []
            for param in self.params:
                oldUpdate = theano.shared(numpy.zeros(param.get_value().shape, dtype=theano.config.floatX))
                oldMeanSquare = theano.shared(numpy.zeros(param.get_value().shape, dtype=theano.config.floatX))
                oldUpdates.append(oldUpdate)
                oldMeanSquares.append(oldMeanSquare)

            deltaParams = T.grad(cost, self.params)
            parametersTuples = zip(self.params,
                                deltaParams,
                                oldUpdates,
                                oldMeanSquares)

            for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
                paramUpdate = momentum * oldUpdate
                meanSquare = rho * oldMeanSquare + (1. - rho) * T.sqr(delta)
                paramUpdate += - (1. - momentum) * learning_rate * delta / T.maximum(T.sqrt(meanSquare), epsilon)
                updates.append((oldMeanSquare, meanSquare))
                newParam = param + paramUpdate
                updates.append((param, newParam))
                updates.append((oldUpdate, paramUpdate))

            return updates

        def classicalMomentum(cost, learning_rate):
            # We must not compute the gradient through the gibbs sampling
            gparams = T.grad(cost, self.params)

            # ... and allocate mmeory for momentum'd versions of the gradient
            gparams_mom = []
            for param in self.params:
                gparam_mom = theano.shared(numpy.zeros(param.get_value().shape, dtype=theano.config.floatX))
                gparams_mom.append(gparam_mom)

            # Update the step direction using momentum
            updates = OrderedDict()
            for gparam_mom, gparam in zip(gparams_mom, gparams):
                # change the update rule to match Hinton's dropout paper
                updates[gparam_mom] = momentum * gparam_mom - (1. - momentum) * gparam * learning_rate

            # ... and take a step along that direction
            for param, gparam_mom in zip(self.params, gparams_mom):
                # since we have included learning_rate in gparam_mom, we don't need it
                # here
                updates[param] = param + updates[gparam_mom]

            return updates

        if self.rmsprop:
            updates = rmsprop(cost=train_cost, learning_rate=T.cast(lr, dtype=theano.config.floatX))
        else:
            updates = classicalMomentum(cost=train_cost, learning_rate=T.cast(lr, dtype=theano.config.floatX))

        train_fn = theano.function(
            inputs=[index],
            outputs=train_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        test_score_i = theano.function(
            inputs=[index],
            outputs=errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        valid_score_i = theano.function(
            inputs=[index],
            outputs=errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        test_initial_i = theano.function(
            inputs=[index],
            outputs=self.negative_log_likelihood(self.y),
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def visualize_features(rbm_weights):
    """ visualize the first 64 filters of a given RBM

    NOTE: The number of visible units must a value with an
    integer-valued square root, and the number of hidden units
    must be at least 64 """

    import matplotlib.pylab as plt
    n_inputs = rbm_weights.shape[0]
    image_size = int(numpy.sqrt(n_inputs))

    for i in range(64):
        feature = numpy.reshape(rbm_weights[:,i],(image_size,image_size))
        plt.subplot(10,10,i)
        plt.imshow(feature)

    plt.show()


def test_DBN(finetune_lr=0.1, pretraining_epochs=200, 
             pretrain_lrs=[0.005, 0.005], k=1, training_epochs=1800,
             hidden_layers_sizes=[800, 800], dropout_rates=[0.8, 0.5, 0.5], 
             pretrain_moms=[0.95, 0.95], finetune_mom=0.8,
             dataset='/mnt/Drive2/reuben/mnist/data/mnist.pkl.gz', batch_size=128):

    """
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

    #dataset = '/Users/reuben_feinman/Data/mnist/mnist.pkl.gz' # local macbook
    #dataset='/home/rfeinman/data/mnist.pkl.gz' # ibil server
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value().shape[0] / batch_size
    n_test_samples = test_set_x.get_value().shape[0]
    n_valid_samples = valid_set_x.get_value().shape[0]

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network

    dbn = DBN(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=hidden_layers_sizes,
              dropout_rates=dropout_rates,
              n_outs=10)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            time0 = time.time()
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index, lr=pretrain_lrs[i], momentum=pretrain_moms[i]))
            total_cost = numpy.mean(c)
            print 'Pre-training layer %i, secs %f, epoch %d, cost ' % (i, time.time() - time0, epoch), total_cost

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    """
    # visually examin features
    weights0 = dbn.rbm_layers[0].W.get_value()
    visualize_features(weights0)
    """

    ########################
    # FINETUNING THE MODEL #
    ########################

    learning_rate = theano.shared(value=numpy.asarray(finetune_lr, dtype=theano.config.floatX),
                                name='learning_rate',borrow=True)
    decay_learning_rate = theano.function(inputs=[], outputs=[],
                                        updates={learning_rate: learning_rate * 0.998})
    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        lr=learning_rate,
        mom=finetune_mom
    )

    print '... finetuning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    #validation_frequency = min(n_train_batches, patience / 2)
    validation_frequency = n_train_batches
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0


    while (epoch < training_epochs):
        if epoch % 20 == 10:
            print(('    current best test error: %f, achieved at epoch %i, '
                'current learning rate: %f') %
                (float(best_test_error_sum)/n_test_samples, best_epoch, learning_rate.get_value().item()))
        epoch = epoch + 1
        time0 = time.time()
        decay_learning_rate()
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.sum(validation_losses)


                #test it on the test set
                test_losses = test_model()
                test_error_sum = numpy.sum(test_losses)
                print(
                    'epoch %i, secs %f, minibatch %i/%i, test error %f, validation error %f'
                    % (
                        epoch,
                        time.time() - time0,
                        minibatch_index + 1,
                        n_train_batches,
                        float(test_error_sum)/n_test_samples,
                        float(this_validation_loss)/n_valid_samples
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_test_error_sum = test_error_sum
                    best_epoch = epoch

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           float(test_error_sum)/n_test_samples))

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score obtained at epoch %i, '
            'with test performance %f'
        ) % (best_epoch, float(best_test_error_sum)/n_test_samples)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))


if __name__ == '__main__':
    test_DBN()
