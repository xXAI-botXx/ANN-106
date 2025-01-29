"""
This file contains different strategies to update the weights of artificial neural networks.

It is important to notice that the output from the neural network depends on the
setted bools in the 'prediction_elements_tuple' variable in :class:~ann.base.ArtificialNeuralNetwork.
"""

import numpy as np
from .base import ArtificialNeuralNetwork


#############
### Tools ###
#############

# -> probably now in loss-function!!!
def build_gradient(func:callable):
    """
    Builds
    """
    pass



########################
### Learn Algorithms ###
########################

def simple_weights_update(network:ArtificialNeuralNetwork, prediction_element):
        """
        A simple learning algorithm for ONLY simple tasks and networks.

        Important setting in :class:~ann106.base.ArtificialNeuralNetwork

        network.prediction_elements_tuple = {
            "X": True,
            "y": True,
            "y_": False,
            "all_y_": True,
            "error": False
        }
        """
        # extract needed elements
        cur_X, cur_y, cur_y_pred = prediction_element 
        cur_y_pred = cur_y_pred[0]
        cur_error = cur_y - cur_y_pred
        
        delta_weights = network.get_lr() * cur_error * cur_X
        network.layers[0].weights = network.layers[0].weights+delta_weights

        network.layers[0].bias = network.layers[0].bias + network.get_lr() * cur_error

def backpropagation(network:ArtificialNeuralNetwork, prediction_element):
    """
    The backpropagation algorithm tries to find a weight build with the gradient of 0,
    which is the global or local minima of the loss-function.

    A gradient is just a vector which points at the biggest slope of a function on a given point and is the first derivative.
    The negative gradient is a vector which points in the direction of the biggest slope in negative direction.
    Gradients are showed with the nabla sign.

    Process:
    1. Start at a random weight -> already happened at the neginning of a network
    2. Calculate the negative gradient of this point -> the negative gradient points at the local or global minima
    3. Adjust the Weight to go in this direction with a given stepsize -> learning rate -> how?
    4. Repeat step 2 and 3 until the negative gradient of the loss function is 0 (or it is near by the solution, or max steps are over)
    """
    # loss_gradient = network.
    pass


