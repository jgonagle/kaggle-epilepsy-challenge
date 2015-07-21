import numpy as np
import theano
import theano.tensor as T

import log_reg as lr

from dropout import dropoutify
from relu import relu, noisy_relu

class HiddenLayer(object):
    
    def __init__(self, rng, srng, input_train, input_valid, num_inputs=128, num_outputs=128, weights_mean_init=0, weights_std_init=.01, biases_init=1, activation=relu, dropout_on=True):
        
        #intialize values of weights and bias
        weights_init_vals = rng.normal(loc=weights_mean_init, scale=weights_std_init, size=(num_inputs, num_outputs))
        biases_init_vals = np.full(num_outputs, biases_init)
        
        self.weights = theano.shared(value=np.asarray(weights_init_vals, dtype=theano.config.floatX), name='hidden layer weights', borrow=True)
        self.biases = theano.shared(value=np.asarray(biases_init_vals, dtype=theano.config.floatX), name='hidden layer biases', borrow=True)
        
        activation_input_train = T.dot(input_train, self.weights) + self.biases
        activation_input_valid = T.dot(input_valid, self.weights) + self.biases
        
        #transform hidden layer activation into monte carlo dropout activation and marginal approximation dropout activation
        activation_train = dropoutify(srng, activation, True, dropout_on)
        activation_valid = dropoutify(srng, activation, False, dropout_on)
        
        self.output_train = activation_train(activation_input_train)
        self.output_valid = activation_valid(activation_input_valid)
        
        self.params = [self.weights, self.biases]

class MLPLayer(object):
    
    def __init__(self, rng, srng, input_train, input_valid, num_inputs=128, num_hidden=128, num_outputs=2, hidden_weights_mean_init=0, output_weights_mean_init=0, hidden_weights_std_init=.01, output_weights_std_init=.01, hidden_biases_init=1, output_biases_init=0, hidden_activation=relu, output_activation=T.nnet.softmax, dropout_on=True):

        self.hidden_layer = HiddenLayer(rng=rng, srng=srng, input_train=input_train, input_valid=input_valid, num_inputs=num_inputs, num_outputs=num_hidden, weights_mean_init=hidden_weights_mean_init, weights_std_init=hidden_weights_std_init, biases_init=hidden_biases_init, activation=hidden_activation, dropout_on=dropout_on)
        
        self.log_reg_layer = lr.LogRegLayer(rng=rng, input_train=self.hidden_layer.output_train, input_valid=self.hidden_layer.output_valid, num_inputs=num_hidden, num_outputs=num_outputs, weights_mean_init=output_weights_mean_init, weights_std_init=output_weights_std_init, biases_init=output_biases_init, activation=output_activation)
        
        self.output_train = self.log_reg_layer.output_train
        self.output_valid = self.log_reg_layer.output_valid
        
        self.valid_label_pred = self.log_reg_layer.valid_label_pred
        
        self.neg_log_like_train = self.log_reg_layer.neg_log_like_train
        self.valid_error = self.log_reg_layer.valid_error
        
        self.params = self.hidden_layer.params + self.log_reg_layer.params