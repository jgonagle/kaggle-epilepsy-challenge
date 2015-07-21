import numpy as np
import theano
import theano.tensor as T

class LogRegLayer(object):
    
    #takes two inputs since previous layers may be dropout layers in which case the monte carlo and marginal approximation computation graphs are needed
    def __init__(self, rng, input_train, input_valid, num_inputs=128, num_outputs=2, weights_mean_init=0, weights_std_init=.01, biases_init=0, activation=T.nnet.softmax):
        
        #intialize values of weights and bias
        weights_init_vals = rng.normal(loc=weights_mean_init, scale=weights_std_init, size=(num_inputs, num_outputs))
        biases_init_vals = np.full(num_outputs, biases_init)
        
        self.weights = theano.shared(value=np.asarray(weights_init_vals, dtype=theano.config.floatX), name='logistic regression weights', borrow=True)
        self.biases = theano.shared(value=np.asarray(biases_init_vals, dtype=theano.config.floatX), name='logistic regression biases', borrow=True)
        
        activation_input_train = T.dot(input_train, self.weights) + self.biases
        activation_input_valid = T.dot(input_valid, self.weights) + self.biases
        
        #activation function for dropout logistic regression is identical (i.e. no dropout)
        self.output_train = activation(activation_input_train)
        self.output_valid = activation(activation_input_valid)
        
        #label with the maximum predictive value per sample
        self.valid_label_pred = T.argmax(self.output_valid, axis=1)
                
        self.params = [self.weights, self.biases]
    
    def neg_log_like_train(self, train_label_true):
    
        #uses numpy/Theanos advanced slice system
        return -T.mean(T.log(self.output_train)[T.arange(train_label_true.shape[0]), train_label_true])
    
    def valid_error(self, valid_label_true):
    
        return T.mean(T.neq(self.valid_label_pred, valid_label_true))