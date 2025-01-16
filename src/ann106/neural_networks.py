

import numpy as np

from .base import ArtificialNeuralNetwork, Layer
from . import loss_functions
from . import activation_functions

def heaviside(sum_value):
    return activation_functions.step_function(sum_value, threshold=0.0, greater_equal_value=1.0, smaller_value=0.0)

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
        cur_prediction_error = loss_functions.get_total_loss(cur_prediction_error) 
        
        delta_weights = cur_prediction_error*cur_X
        self.layers[0].weights = self.layers[0].weights+delta_weights

        self.layers[0].bias = self.layers[0].bias + (cur_prediction_error)

    def loss_function(self, y, y_):
        return {"Sum Loss": loss_functions.sum_error(y, y_)}

    def predict(self, x):
        return self.forward(x)



class Adaline(ArtificialNeuralNetwork):
    def __init__(self):
        super().__init__()
        self.prediction_elements_tuple = {
            "X": True,
            "y": True,
            "y_": False,
            "all_y_": True,
            "error": False
        }
        self.name = "Adaline"
        self.layers = [
                Layer(2, 1, None), 
                    ]

    def update_weights(self, prediction_element):
        # extract needed elements
        cur_X, cur_y, cur_y_pred = prediction_element 
        cur_y_pred = cur_y_pred[0]
        cur_error = cur_y - cur_y_pred
        
        delta_weights = self.get_lr() * cur_error * cur_X
        self.layers[0].weights = self.layers[0].weights+delta_weights

        self.layers[0].bias = self.layers[0].bias + self.get_lr() * cur_error

    def loss_function(self, y, y_):
        # return {"Sum Loss":loss_functions.sum_error(y, y_)} # y - y_
        return {"MSE":loss_functions.mean_squared_error(y, y_)}

    def eval(self, X, y, loss_func=None, group_function=None):

        # Lambda function can't be saved with pickle, so use normal functions
        def mean_squared_root_error(y, y_):
            return np.sqrt(np.mean((y-y_)**2))

        def group_function_(errors):
            return np.mean(errors)

        return super().eval(X, y, loss_func=mean_squared_root_error, group_function=group_function_)

    def predict(self, x):
        x = self.scale(x)
        return activation_functions.step_function(self.forward(x), threshold=0.0, greater_equal_value=1, smaller_value=-1)



