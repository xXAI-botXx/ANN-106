"""

IN PROGRESS...

To Dos:
- Add abstract functions to ANN class
- Add basic methods to ANN clas, which every ANN needs -> general fit method?
- How to build a Neural Network? How to add layers?
- Add batchsize to -> parallel processing + adjust weights after all datapoints in the batch processed
    - Parallel(n_jobs=-1)(delayed(compute_square)(num) for num in numbers
- Add data to train_history after fit is finish
- Add prints -> like from https://github.com/xXAI-botXx/torch-mask-rcnn-instance-segmentation
- Make the Code more general -> less redundance
- Add Sphinx Strings
- GPU Support?
- Dataloader programmieren?

"""

# from copy import deepcopy
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def get_cur_time_str():
    now = datetime.now()
    return f"{now.day}.{now.month}.{now.year} {now.hour}:{now.minute}:{now.second}"


class Layer():
    def __init__(self, input_size, output_size, activation_func):
        """
        Initializes a single layer in the neural network.

        Parameters
        ----------
        input_size : int
            Number of input neurons to the layer.
        output_size : int
            Number of output neurons from the layer.
        activation_func : callable
            Activation function for the layer.
        """
        self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        self.bias = np.random.uniform(-1, 1, (output_size,))
        self.activation_func = activation_func if activation_func else lambda x: x 

    def forward(self, X):
        """
        Forward pass through the layer.

        Parameters
        ----------
        X : np.ndarray
            Input data for the layer.

        Returns
        -------
        np.ndarray
            Output of the layer after applying the activation function.
        """
        z = np.dot(self.weights, X) + self.bias
        return self.activation_func(z)



class Artificial_Neural_Network():
    def __init__(self):
        self.layers = []
        self.name = "ANN"

        # choose which elments you want for your "update_weights" method
        self.prediction_elements_tuple = {
            "X": True,
            "y": False,
            "y_": False,
            "error": True
        }

        self.train_history = {
            "start-time":[],
            "end-time":[],
            "epochs":[],
            "total-steps":[],
            "batch-size":[],
            "all-errors":[]
        }

    def forward(self, X):
        """
        Forward pass through the entire network.

        Parameters
        ----------
        X : np.ndarray
            Input data for the network.

        Returns
        -------
        np.ndarray
            Final output of the network.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, X, y, epochs, batch_size=1, parallel_computing=True, shuffle_data=True):
        """
        Trains the artificial neural network (ANN) on the given data.

        Parameters
        ----------
        X : np.ndarray
            Array with multiple datapoints (features).
        y : np.ndarray
            Array with ground truth labels (targets).

        Returns
        -------
        None
            Updates ANN Model internally
        """
        
        # new_model = deepcopy(self)
        output_print = f"\n{'-'*24}\n{self.name} Training:\n    - Epochs: {epochs}\n    - Layers: {len(self.layers)}"
        
        start_time = get_cur_time_str()
        all_errors = []
        cur_step = 0
        n_samples = X.shape[0]

        # Go through every datapoint X-times
        for cur_epoch in range(epochs):
            cur_iteration = 0

            output_print += f"\n\n{'#'*24}\n{cur_epoch+1}. Epoch:"
            progress = (cur_epoch+1)/epochs
            blocks = 20
            output_print += f"\n    -> Progress: |{'#'*int(blocks*progress)}{' '*int(blocks*(1-progress))}|"

            # shuffle data
            if shuffle_data:
                permutation = np.random.permutation(len(X))
                X_shuffled = X[permutation]
                y_shuffled = y[permutation]
            else:
                X_shuffled = X
                y_shuffled = y

             # Process in batches
            for i in range(0, n_samples, batch_size): 
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                prediction_cache = []

                if parallel_computing:
                    pass
                else:
                    for cur_X, cur_y in zip(X_batch, y_batch):

                        # predict and calc error
                        y_ = self.forward(cur_X)
                        error = self.loss_function(cur_y, y_)    # loss

                        # how to customize prediction_elments
                        prediction_elments = []
                        if self.prediction_elements_tuple["X"]:
                            prediction_elments += [cur_X]
                        if self.prediction_elements_tuple["y"]:
                            prediction_elments += [cur_y]
                        if self.prediction_elements_tuple["y_"]:
                            prediction_elments += [y_]
                        if self.prediction_elements_tuple["error"]:
                            prediction_elments += [error]
                        prediction_cache += [prediction_elments]

                        cur_step += 1

                # adjust weights
                for cur_prediction_element in prediction_cache:
                    self.update_weights(cur_prediction_element)

                all_errors += [error]
        
        end_time = get_cur_time_str()

        self.train_history["start-time"] += [start_time]
        self.train_history["end-time"] += [end_time]
        self.train_history["epochs"] += [cur_epoch]
        self.train_history["total-steps"] += [cur_step]
        self.train_history["batch-size"] += [batch_size]
        self.train_history["all-errors"] += [all_errors]

    def add_layer(self, input_size, output_size, activation_func=None):
        """
        Adds a new layer to the neural network.

        Parameters
        ----------
        input_size : int
            Number of input neurons to the layer.
        output_size : int
            Number of output neurons from the layer.
        activation_func : callable
            Activation function for the layer.

        Returns
        -------
        None
        """
        layer = Layer(input_size, output_size, activation_func)
        self.layers.append(layer)

    def save(self, save_path:str):
        # FIXME
        pass

    def load(save_path:str):
        # FIXME
        pass

    # UPDATE ME
    def update_weights(self, prediction_element):
        pass

    # UPDATE ME
    def loss_function(self, y, y_):
        return y - y_

    # UPDATE ME
    def eval(self, X, y):
        absolute_error = 0

        for i, x in enumerate(X):
            absolute_error += np.abs(y[i] - self.forward(x))

        return absolute_error














