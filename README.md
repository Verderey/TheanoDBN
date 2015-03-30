This library extends the Theano tutorial DBN implementation to include momentum, RMSprop and dropout.

Sparse initialization scheme from section 3.1 of Hinton's paper:
  http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
  
Using theano.sandbox.rng_mrg.MRG_RandomStreams instead of the default Theano RNG because of this issue: https://github.com/Theano/Theano/issues/1233
Pretraining time per epoch improved by a factor of 14 after making this change.
