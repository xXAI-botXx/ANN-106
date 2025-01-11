

import numpy as np

from .base import Artificial_Neural_Network, Layer
from .activation_functions import heaviside
from .loss_functions import sum_error


class Perceptron(Artificial_Neural_Network):
    def __init__(self):
        super().__init__()
        self.prediction_elements_tuple = {
            "X": True,
            "y": False,
            "y_": False,
            "error": True
        }
        self.name = "Perceptron"
        self.layers = [Layer(2, 1, heaviside)]

    def update_weights(self, prediction_element):
        # extract needed elements
        cur_X, cur_prediction_error = prediction_element 
        
        delta_weights = cur_prediction_error*cur_X
        self.layers[0].weights = self.layers[0].weights+delta_weights

        self.layers[0].bias = self.layers[0].bias + (cur_prediction_error)

    def loss_function(self, y, y_):
        return sum_error(y, y_)

    def eval(self, X, y):
        absolute_error = 0

        for i, x in enumerate(X):
            absolute_error += np.abs(y[i] - self.forward(x))

        return absolute_error



