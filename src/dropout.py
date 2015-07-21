import numpy as np
import theano
import theano.tensor as T

two = T.cast(2.0, dtype=theano.config.floatX)

#could be implemented more elegantly using T.switch?
def dropoutify(srng, activation, is_monte_carlo=False, dropout_on=True):
    
    if dropout_on:
        def dropout_activation(activation_input):
            pre_dropout = activation(activation_input)
            
            if is_monte_carlo:
                return pre_dropout * srng.binomial(n=1, p=.5, size=pre_dropout.shape)
            else:
                return pre_dropout / two
        
        return dropout_activation
    else:
        return activation