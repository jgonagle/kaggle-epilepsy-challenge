import numpy as np
import theano

#returns max of zero and activation
def relu(activation_input):
    
    return activation_input * (activation_input > 0)

#returns max of zero and activation + unit gaussian noise 
def noisy_relu(srng, activation_input):
    
    noise = srng.normal(size=activation_input.shape, avg=0.0, std=activation_input)
    noisy_activation_val = activation_input + noise
    
    return relu(noisy_activation)