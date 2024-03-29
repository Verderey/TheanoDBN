"""
Author: 
Reuben Feinman
Brown University '15

Stencil code provided by www.deeplearning.com/tutorial/code
"""

import time
import os
import random

import numpy

import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams


class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None,
        rmsprop=False
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.rmsprop = rmsprop

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        '''
        Sparse initialization scheme from section 3.1 of Hinton's paper:
        http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
        '''
        num_connections = 10
        scale = 0.8

        if W is None:
            indices = range(n_visible)
            weights = numpy.zeros((n_visible, n_hidden),dtype=theano.config.floatX)
            for i in range(n_hidden):
                random.shuffle(indices)
                for j in indices[:num_connections]:
                    weights[j,i] = random.gauss(0.0, scale)

            W = theano.shared(value=weights, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias'
            )

        if vbias is None:
            ### SS initialization scheme
            visible_biases = 0.01 * numpy.random.randn(n_visible,)
            visible_biases = visible_biases.astype(theano.config.floatX)
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=visible_biases,
                name='vbias'
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1, mom=0.9):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """
        momentum = T.cast(mom, dtype=theano.config.floatX)
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        rbm_cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))

        def rmsprop(cost, learning_rate, updates, r=0.9, epsilon=1e-2):
            rho = T.cast(r, dtype=theano.config.floatX)
            rFactor = T.cast(1.0, dtype=theano.config.floatX) - rho
            lrFactor = T.cast(1.0, dtype=theano.config.floatX) - momentum

            oldUpdates = []
            oldMeanSquares = []
            for param in self.params:
                oldUpdate = theano.shared(numpy.zeros(param.get_value().shape, dtype=theano.config.floatX))
                oldMeanSquare = theano.shared(numpy.zeros(param.get_value().shape, dtype=theano.config.floatX))
                oldUpdates.append(oldUpdate)
                oldMeanSquares.append(oldMeanSquare)

            deltaParams = T.grad(cost, self.params, consider_constant=[chain_end])
            parametersTuples = zip(self.params,
                                deltaParams,
                                oldUpdates,
                                oldMeanSquares)

            for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
                paramUpdate = momentum * oldUpdate
                meanSquare = rho * oldMeanSquare + rFactor * T.sqr(delta)
                paramUpdate += - lrFactor * learning_rate * delta / T.maximum(T.sqrt(meanSquare), epsilon)
                updates[oldMeanSquare] = meanSquare
                newParam = param + paramUpdate
                updates[param] = newParam
                updates[oldUpdate] = paramUpdate

        def classicalMomentum(cost, learning_rate, updates):
            # We must not compute the gradient through the gibbs sampling
            gparams = T.grad(cost, self.params, consider_constant=[chain_end])

            # ... and allocate mmeory for momentum'd versions of the gradient
            gparams_mom = []
            for param in self.params:
                gparam_mom = theano.shared(numpy.zeros(param.get_value().shape, dtype=theano.config.floatX))
                gparams_mom.append(gparam_mom)

            # Update the step direction using momentum
            for gparam_mom, gparam in zip(gparams_mom, gparams):
                # change the update rule to match Hinton's dropout paper
                updates[gparam_mom] = momentum * gparam_mom - (1. - momentum) * gparam * learning_rate

            # ... and take a step along that direction
            for param, gparam_mom in zip(self.params, gparams_mom):
                # since we have included learning_rate in gparam_mom, we don't need it
                # here
                updates[param] = param + updates[gparam_mom]

        if self.rmsprop:
            rmsprop(cost=rbm_cost, learning_rate=T.cast(lr, dtype=theano.config.floatX), updates=updates)
        else:
            classicalMomentum(cost=rbm_cost, learning_rate=T.cast(lr, dtype=theano.config.floatX), updates=updates)

        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy


class GBRBM(RBM):

    # Implementation of Gaussian-Bernoulli RBM. Will need to use fewer training epochs
    def __init__(self, input=None, n_visible=784, n_hidden=500, 
                W=None, hbias=None, vbias=None, numpy_rng=None, 
                theano_rng=None, rmsprop=True):
        # initialize parent class (RBM)
        RBM.__init__(self, input=input, n_visible=n_visible, n_hidden=n_hidden,
                    W=W, hbias=hbias, vbias=vbias, numpy_rng=numpy_rng, 
                    theano_rng=theano_rng, rmsprop=rmsprop)

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = 0.5*T.sum(T.sqr(v_sample - self.vbias), axis=1)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.normal(size=v1_mean.shape, avg=v1_mean, std=1.0, dtype=theano.config.floatX) + pre_sigmoid_v1
        # use this instead if data is normalized to zero mean and 1 std
        #v1_sample = pre_sigmoid_v1
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        # cross-entropy will not be a good evaluation technique in this case;
        # use squared error
        cost = T.sum(T.sqr(self.input - pre_sigmoid_nv))
        return cost

class DropoutRBM(RBM):

    # --------------------------------------------------------------------------
    # initialize class
    def __init__(self, input=None, n_visible=784, n_hidden=500, 
                W=None, hbias=None, vbias=None, numpy_rng=None, 
                theano_rng=None, rmsprop=True, hiddenDropout=0.5):

        # initialize parent class (RBM)
        RBM.__init__(self, input=input, n_visible=n_visible, n_hidden=n_hidden,
                    W=W, hbias=hbias, vbias=vbias, numpy_rng=numpy_rng, 
                    theano_rng=theano_rng, rmsprop=rmsprop)

        self.hiddenDropout = hiddenDropout

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        drop_mask = self.theano_rng.binomial(n=1, p=self.hiddenDropout, 
                                            size=h1_mean.shape, dtype=theano.config.floatX)
        h1_mean = h1_mean * drop_mask
        h1_sample = h1_sample * drop_mask
        return [pre_sigmoid_h1, h1_mean, h1_sample]
