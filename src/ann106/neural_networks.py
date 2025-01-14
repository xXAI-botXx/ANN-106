

import numpy as np

from .base import ArtificialNeuralNetwork, Layer
from .activation_functions import step_function
from .loss_functions import sum_error, get_total_loss

def heaviside(sum_value):
    return step_function(sum_value, threshold=0.0, greater_equal_value=1.0, smaller_value=0.0)

class Perceptron(ArtificialNeuralNetwork):
    def __init__(self):
        super().__init__()
        self.prediction_elements_tuple = {
            "X": True,
            "y": False,
            "y_": False,
            "all_y_": False,
            "error": True
        }
        self.name = "Perceptron"
        self.layers = [Layer(2, 1, heaviside)]

    def update_weights(self, prediction_element):
        # extract needed elements
        cur_X, cur_prediction_error = prediction_element
        cur_prediction_error = get_total_loss(cur_prediction_error) 
        
        delta_weights = cur_prediction_error*cur_X
        self.layers[0].weights = self.layers[0].weights+delta_weights

        self.layers[0].bias = self.layers[0].bias + (cur_prediction_error)

    def loss_function(self, y, y_):
        return {"Sum Loss": sum_error(y, y_)}

    def predict(self, x):
        return x

    def eval(self, X, y):
        absolute_error = 0

        for i, x in enumerate(X):
            absolute_error += np.abs(y[i] - self.forward(x))

        return absolute_error



