import sys
import time
import math
import numpy as np
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.nnet import sigmoid
from theano.tensor.signal.downsample import max_pool_2d

import mlp

from dropout import dropoutify
from relu import relu, noisy_relu

#theano requires convolution and pooling shapes at compile time
num_conv_layers = 2
batch_size = 500
conv_image_dims = [(28, 28), (12, 12)]
conv_filter_dims = [(4, 4), (4, 4)]
conv_max_pool_dims=[(2, 2), (2, 2)]
mlp_image_dims = (4, 4)

save_every_n_epochs = 5
            
yes_set = set(['y', 'yes', ''])
no_set = set(['n', 'no'])

class ConvLayer(object):

    def __init__(self, rng, srng, image_batch_train, image_batch_valid, image_dims, filter_dims, max_pool_dims, weights_mean_init=0, weights_std_init=.01, biases_init=1, activation=relu, dropout_on=True):
        """
        
        filter_dims = (number of filters, num image features, filter height, filter width)
        image_dims = (batch size, num image features, image height, image width)
        """
        
        #intialize values of weights and bias
        weights_init_vals = rng.normal(loc=weights_mean_init, scale=weights_std_init, size=filter_dims)
        biases_init_vals = np.full(filter_dims[0], biases_init)
        
        self.weights = theano.shared(value=np.asarray(weights_init_vals, dtype=theano.config.floatX), name='conv layer weights', borrow=True)
        self.biases = theano.shared(value=np.asarray(biases_init_vals, dtype=theano.config.floatX), name='conv layer biases', borrow=True)
        
        #determine convolutional layer outgoing values sans the biases (added later after the max pooling step)
        conv_output_train = conv2d(input=image_batch_train, filters=self.weights, filter_shape=filter_dims, image_shape=image_dims)
        conv_output_valid = conv2d(input=image_batch_valid, filters=self.weights, filter_shape=filter_dims, image_shape=image_dims)

        #max pool over filter activations according to the max pool dimensions
        max_pool_output_train = max_pool_2d(input=conv_output_train, ds=max_pool_dims, ignore_border=True)
        max_pool_output_valid = max_pool_2d(input=conv_output_valid, ds=max_pool_dims, ignore_border=True)
        
        #add in the bias
        shuffled_biases = self.biases.dimshuffle('x', 0, 'x', 'x')
        activation_input_train = max_pool_output_train + shuffled_biases
        activation_input_valid = max_pool_output_valid + shuffled_biases       
        
        #transform layer activation into monte carlo dropout activation and marginal approximation dropout activation
        activation_train = dropoutify(srng, activation, True, dropout_on)
        activation_valid = dropoutify(srng, activation, False, dropout_on)
        
        self.output_train = activation_train(activation_input_train)
        self.output_valid = activation_valid(activation_input_valid)

        #layer parameters
        self.params = [self.weights, self.biases]

class ConvNNet(object):
    
    #vector_format is a boolean determining whether each image is encoded as a vector or a 2D matrix
    def __init__(self, vector_format=False, num_conv_features=[16, 64], num_mlp_hidden=256, num_outputs=2, conv_weights_mean_init=[0, 0], hidden_weights_mean_init=0, output_weights_mean_init=0, conv_weights_std_init=[.01, .01], hidden_weights_std_init=.01, output_weights_std_init=.01, conv_biases_init=[0, 0], hidden_biases_init=0, output_biases_init=0, conv_activation=[relu, relu], hidden_activation=relu, output_activation=T.nnet.softmax, dropout_on=True):
        
        print(''.join(['\nBeginning initialization of convolutional neural net with ', str(num_conv_layers), ' convolutional layers and one ', str(num_mlp_hidden), ' feature mlp layer']))
        
        self.num_outputs = num_outputs
        
        #input image batch
        if vector_format:
            self.image_batch = T.matrix('image batch')
        else:
            self.image_batch = T.tensor3('image batch')
        
        #image label batch
        self.label_batch = T.lvector('label batch')
        
        rng = np.random.RandomState()
        srng = RandomStreams()
        
        #store layers used for computation
        self.conv_layers = [None for i in range(num_conv_layers)]
        
        num_prev_features = 1
        prev_layer_output_train = self.image_batch.reshape((batch_size, num_prev_features, conv_image_dims[0][0], conv_image_dims[0][1]))
        prev_layer_output_valid = self.image_batch.reshape((batch_size, num_prev_features, conv_image_dims[0][0], conv_image_dims[0][1]))
        
        #create each layer incrementally and connect outputs from previous layer to inputs of next layer
        for i in range(num_conv_layers):
            #same for training and valid outputs
            num_features = num_conv_features[i]
            image_dims = (batch_size, num_prev_features, conv_image_dims[i][0], conv_image_dims[i][1])
            filter_dims = (num_features, num_prev_features, conv_filter_dims[i][0], conv_filter_dims[i][1])
            max_pool_dims = conv_max_pool_dims[i]
            weights_mean_init = conv_weights_mean_init[i]
            weights_std_init = conv_weights_std_init[i]
            biases_init = conv_biases_init[i]
            activation = conv_activation[i]
                                    
            self.conv_layers[i] = ConvLayer(rng, srng, image_batch_train=prev_layer_output_train, image_batch_valid=prev_layer_output_valid, image_dims=image_dims, filter_dims=filter_dims, max_pool_dims=max_pool_dims, weights_mean_init=weights_mean_init, weights_std_init=weights_std_init, biases_init=biases_init, activation=activation, dropout_on=dropout_on)
            
            num_prev_features = num_features
            prev_layer_output_train = self.conv_layers[i].output_train
            prev_layer_output_valid = self.conv_layers[i].output_valid
        
        mlp_input_train = prev_layer_output_train.flatten(2)
        mlp_input_valid = prev_layer_output_valid.flatten(2)
        #dimensions to hidden layer from flattened output of convolutional layers
        mlp_num_inputs = np.prod(mlp_image_dims) * num_prev_features
        
        self.mlp_layer = mlp.MLPLayer(rng, srng, input_train=mlp_input_train, input_valid=mlp_input_valid, num_inputs=mlp_num_inputs, num_hidden=num_mlp_hidden, num_outputs=num_outputs, hidden_weights_mean_init=hidden_weights_mean_init, output_weights_mean_init=output_weights_mean_init, hidden_weights_std_init=hidden_weights_std_init, output_weights_std_init=output_weights_std_init, hidden_biases_init=hidden_biases_init, output_biases_init=output_biases_init, hidden_activation=hidden_activation, output_activation=output_activation, dropout_on=dropout_on)

        self.output_train = self.mlp_layer.output_train
        self.output_valid = self.mlp_layer.output_valid
        
        self.valid_label_pred = self.mlp_layer.valid_label_pred
        
        conv_params = [p for cl in self.conv_layers for p in cl.params]
        mlp_params = self.mlp_layer.params
        #combine model params into one list to be used for optimization
        self.params = conv_params + mlp_params
        self.params_change_last = [theano.shared(value=np.zeros_like(p.get_value(), dtype=theano.config.floatX), name=p.name+' change last', borrow=True) for p in self.params]
        
        print('Finished initialization of convolutional neural net')
    
    def train(self, labeled_data, is_gen=False, valid_frac=None, learning_rate=.1, momentum=.8, weight_decay=0, num_epochs=100, save_func=None, save_args=[], save_on=False):
        
        print(''.join(['\nBeginning training with learning rate of ', str(learning_rate), ', momentum of ', str(momentum), ', weight decay of ', str(weight_decay), ', and batch size of ', str(batch_size), ' for ', str(num_epochs), ' epochs']))
        
        learning_rate = T.cast(learning_rate, dtype=theano.config.floatX)
        momentum = T.cast(momentum, dtype=theano.config.floatX)
        weight_decay = T.cast(weight_decay, dtype=theano.config.floatX)
        
        #cost to optimize is the negative log likelihood of the labels given the model output on the training data
        cost = self.mlp_layer.neg_log_like_train(self.label_batch)
        #gradient of the negative log likelihood with respect to the model parameters
        grad = T.grad(cost, self.params)
        
        #how to update the parameters after each round of batch training
        param_changes_cur = [(momentum * p_change_last - learning_rate * g - weight_decay * p_cur) for p_cur, p_change_last, g in zip(self.params, self.params_change_last, grad)]
        
        updates = [(p_cur, p_cur + p_change_cur) for p_cur, p_change_cur in zip(self.params, param_changes_cur)]
        updates.extend([(p_change_last, p_change_cur) for p_change_last, p_change_cur in zip(self.params_change_last, param_changes_cur)])
        
        if not is_gen:
            num_samples = len(labeled_data)
            
            size_valid_pool = math.ceil(num_samples * valid_frac)
            size_train_pool = num_samples - size_valid_pool
            
            np.random.shuffle(labeled_data)
            
            valid_images_labels_pool = labeled_data[0:size_valid_pool]
            train_images_labels_pool = labeled_data[size_valid_pool:]
            
            num_valid_batches = size_valid_pool // batch_size
            num_train_batches = size_train_pool // batch_size
        else:
            valid_images_labels_pool, train_images_labels_pool = labeled_data.__next__()
            
            num_valid_batches = len(valid_images_labels_pool) // batch_size
            num_train_batches = len(train_images_labels_pool) // batch_size
        
        num_valid_samples = num_valid_batches * batch_size
        num_train_samples = num_train_batches * batch_size
            
        best_v_error = np.inf
        v_error = np.inf
        
        epoch = 0
        iteration = 0
        
        #keep track of which batch we're training on per epoch
        batch_index = T.lscalar('batch index')
        
        start_time = time.clock()
        
        try:
            while (epoch < num_epochs):
                epoch += 1
                
                #if labeled_data is a generator, get the next set of training data (usually picked at random from some larger training set, according to a true label distribution)
                if is_gen:
                    valid_images_labels_pool, train_images_labels_pool = labeled_data.__next__()
                
                np.random.shuffle(valid_images_labels_pool)
                np.random.shuffle(train_images_labels_pool)
                
                valid_images_labels = valid_images_labels_pool[0:num_valid_samples]
                train_images_labels = train_images_labels_pool[0:num_train_samples]
            
                valid_images_value, valid_labels_value = list(zip(*valid_images_labels))
                train_images_value, train_labels_value = list(zip(*train_images_labels))               
                
                valid_images = theano.shared(value=np.asarray(valid_images_value, dtype=theano.config.floatX), name='valid images', borrow=True)
                train_images = theano.shared(value=np.asarray(train_images_value, dtype=theano.config.floatX), name='train images', borrow=True)
                                
                valid_labels = theano.shared(value=np.asarray(valid_labels_value, dtype='int64'), name='valid labels', borrow=True)
                train_labels = theano.shared(value=np.asarray(train_labels_value, dtype='int64'), name='train labels', borrow=True)
                
                #calculate validation error on batch
                valid_error = theano.function([batch_index], self.mlp_layer.valid_error(self.label_batch), givens={self.image_batch: valid_images[batch_index * batch_size: (batch_index + 1) * batch_size], self.label_batch: valid_labels[batch_index * batch_size: (batch_index + 1) * batch_size]})
                
                #calculate the cost so that we can take the gradient for training
                train_params = theano.function([batch_index], cost, updates=updates, givens={self.image_batch: train_images[batch_index * batch_size: (batch_index + 1) * batch_size], self.label_batch: train_labels[batch_index * batch_size: (batch_index + 1) * batch_size]})
                
                #calculate the output of the convolutional neural net on the training data
                #cnn_output_train = theano.function([batch_index], self.output_train, givens={self.image_batch: train_images[batch_index * batch_size: (batch_index + 1) * batch_size]})
                
                for train_batch_index in range(num_train_batches):
                    #print the probability of the true label for each image in this batch
                    #cot = cnn_output_train(train_batch_index)
                    #print([cot[i][train_labels_shuf_value[train_batch_index * batch_size + i]] for i in range(batch_size)])
                    
                    batch_cost = train_params(train_batch_index)
                
                v_error = np.mean([valid_error(valid_batch_index) for valid_batch_index in range(num_valid_batches)])
                
                print(''.join(['\tEpoch ', str(epoch), ' training complete with validation error of ', str(v_error)]))

                #update best valid error (if applicable) and possibly save model
                if v_error < best_v_error:
                    #update new best validation error
                    best_v_error = v_error
                    #save the new best model
                    if save_on:
                        self.save_self(learning_rate, momentum, weight_decay, best_v_error, save_func, save_args)
                elif save_on and epoch % save_every_n_epochs == 0:
                    self.save_self(learning_rate, momentum, weight_decay, v_error, save_func, save_args)
        
        #on Ctrl-C, give the option to save the current model, then exit              
        except KeyboardInterrupt:
            if save_on:
                choice = '?'
                
                #until a valid choice has been made, prompt for a yes or no answer on whether to save
                while (choice not in yes_set) and (choice not in no_set):
                    choice = input("\nCtrl-C was pressed.  Do you wish to save? (Y/N):").lower()
                
                #save the current model if the choice was yes
                if choice in yes_set:
                    self.save_self(learning_rate, momentum, weight_decay, v_error, save_func, save_args)
                                                                    
            sys.exit(0)

        end_time = time.clock()
        
        print(''.join(['Training for ', str(num_epochs), ' epochs completed in ', str((end_time - start_time) / 60), ' minutes with best validation error of ', str(best_v_error)]))
    
    def test_distribution(self, test_data, save_func=None, save_args=[], save_on=False):
        
        num_test_samples = len(test_data)
        num_test_batches = math.ceil(num_test_samples / batch_size)
        num_padded_test_samples = num_test_batches * batch_size
        pad_length = num_padded_test_samples - num_test_samples
        
        padded_test_data = np.vstack([np.asarray(test_data), np.zeros((pad_length,) + test_data[0].shape)])
        
        print(''.join(['\nBeginning calculation of test label distributions with batch size of ', str(batch_size), ' for ', str(num_test_batches), ' batches']))
        
        test_images = theano.shared(value=np.asarray(padded_test_data, dtype=theano.config.floatX), name='test images', borrow=True)
        
        #keep track of which batch we're training on per epoch
        batch_index = T.lscalar('batch index')
        
        #determine the most likely label for each test sample
        test_label_dist = theano.function([batch_index], self.output_valid, givens={self.image_batch: test_images[batch_index * batch_size: (batch_index + 1) * batch_size]})
                
        all_test_distributions = -np.ones((num_padded_test_samples, self.num_outputs))
        
        start_time = time.clock()
        
        try:            
            for test_batch_index in range(num_test_batches):
                
                test_batch_distributions = test_label_dist(test_batch_index)
                all_test_distributions[test_batch_index * batch_size: (test_batch_index + 1) * batch_size] = test_batch_distributions          
        #on Ctrl-C, give the option to save the current test label distributions, then exit              
        except KeyboardInterrupt:
            if save_on:
                choice = '?'
                
                #until a valid choice has been made, prompt for a yes or no answer on whether to save
                while (choice not in yes_set) and (choice not in no_set):
                    choice = input("\nCtrl-C was pressed.  Do you wish to save? (Y/N):").lower()
                
                #save the current test label distributions if the choice was yes
                if choice in yes_set:
                    save_func(all_test_distributions, *save_args)
                                                                    
            sys.exit(0)
        
        all_test_distributions = all_test_distributions[0:num_test_samples]
        
        end_time = time.clock()
        
        if save_on:
            save_func(all_test_distributions, *save_args)
    
        print(''.join(['\Calculation of test label distributions with batch size of ', str(batch_size), ' for ', str(num_test_batches), ' batches completed in ', str((end_time - start_time) / 60), ' minutes']))
        
        return all_test_distributions
    
    def test_prediction(self, test_data, save_func=None, save_args=[], save_on=False):
        
        num_test_samples = len(test_data)
        num_test_batches = math.ceil(num_test_samples / batch_size)
        num_padded_test_samples = num_test_batches * batch_size
        pad_length = num_padded_test_samples - num_test_samples
        
        padded_test_data = np.vstack([np.asarray(test_data), np.zeros((pad_length,) + test_data[0].shape)])
        
        print(''.join(['\nBeginning calculation of test label predictions with batch size of ', str(batch_size), ' for ', str(num_test_batches), ' batches']))
        
        test_images = theano.shared(value=np.asarray(padded_test_data, dtype=theano.config.floatX), name='test images', borrow=True)
        
        #keep track of which batch we're testing
        batch_index = T.lscalar('batch index')
                
        #determine the most likely label for each test sample
        test_label_pred = theano.function([batch_index], self.valid_label_pred, givens={self.image_batch: test_images[batch_index * batch_size: (batch_index + 1) * batch_size]})
        
        all_test_predictions = [-1 for p in range(num_padded_test_samples)]
        
        start_time = time.clock()
                
        try:     
            for test_batch_index in range(num_test_batches):
                
                test_batch_predictions = test_label_pred(test_batch_index)
                all_test_predictions[test_batch_index * batch_size: (test_batch_index + 1) * batch_size] = test_batch_predictions            
        #on Ctrl-C, give the option to save the current test label predictions, then exit              
        except KeyboardInterrupt:
            if save_on:
                choice = '?'
                
                #until a valid choice has been made, prompt for a yes or no answer on whether to save
                while (choice not in yes_set) and (choice not in no_set):
                    choice = input("\nCtrl-C was pressed.  Do you wish to save? (Y/N):").lower()
                
                #save the current test label predictions if the choice was yes
                if choice in yes_set:
                    save_func(all_test_predictions, *save_args)
                                                                    
            sys.exit(0)
        
        all_test_predictions = all_test_predictions[0:num_test_samples]
        
        end_time = time.clock()
        
        if save_on:
            save_func(all_test_predictions, *save_args)
    
        print(''.join(['\Calculation of test label predictions with batch size of ', str(batch_size), ' for ', str(num_test_batches), ' batches completed in ', str((end_time - start_time) / 60), ' minutes']))
        
        return all_test_predictions
    
    def save_self(self, learning_rate, momentum, weight_decay, v_error, save_func, save_args):
        
        save_func(self, [learning_rate, momentum, weight_decay, v_error], *save_args)

def calc_cnn_image_dims(start_image_dims, conv_filter_dims, conv_max_pool_dims):
    
    num_conv_layers = min(len(conv_filter_dims), len(conv_max_pool_dims))
    
    conv_image_dims = [None for i in range(num_conv_layers + 1)]
    conv_image_dims[0] = start_image_dims
    
    for i in range(1, num_conv_layers + 1):
        filter_height = conv_filter_dims[i - 1][0]
        filter_width = conv_filter_dims[i - 1][1]
        max_pool_height = conv_max_pool_dims[i - 1][0]
        max_pool_width = conv_max_pool_dims[i - 1][1]
        prev_image_height = conv_image_dims[i - 1][0]
        prev_image_width = conv_image_dims[i - 1][1]
                       
        conv_image_dims[i] = ((prev_image_height - filter_height + 1) // max_pool_height, (prev_image_width - filter_width + 1) // max_pool_width)
    
    return conv_image_dims