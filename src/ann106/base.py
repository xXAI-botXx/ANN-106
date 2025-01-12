"""
A simple Framework for artificial neural networks. With prints, logs and more.
It is recommended to use PyTorch or TensorFlow for professional projects, this is just an educational and fun project.

Features
--------
- Weight representation through layers (Layer class)
- Customizable ANN
- Multiple Loss functions
- Sklearn Version
- Train History (Time, Errors)
- Lossplotting
- Added Batch-Size and general training loop

Planned Features
----------------

To Dos:
- Add backward -> learning
- Add parallel processing (of batches)
    - Parallel(n_jobs=-1)(delayed(compute_square)(num) for num in numbers
- Add more Document Strings -> see: https://github.com/xXAI-botXx/Project-Helper/blob/main/guides/Sphinx_Helper.md#How-you-should-code
- Add GPU calculation
- Add Dataloader (?)
- Add more networks -> CNN, ...

"""

#############################
########## Imports ##########
#############################

# from copy import deepcopy
import sys
import os
import pickle
from datetime import datetime, timedelta
import time

from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd






#############################
##### Helper Functions ######
#############################

def get_cur_time_str():
    """
    Returns the current time and date as string: day.month.year hour:minute:second
    """
    now = datetime.now()
    return f"{now.day}.{now.month}.{now.year} {now.hour}:{now.minute}:{now.second}"

def moving_average_sliding_window(data:list, window_size):
    """
    Smooths a list moving a window over the data and applying the average on the window.
    """
    data = np.array(data)
    windows = np.lib.stride_tricks.sliding_window_view(data, window_size)
    return np.mean(windows, axis=1)

def clear_printing():
    """
    Clears the output printing.
    """
    # terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # notebook
    clear_output()

def log(file_path, content, reset_logs=False, reset_output=False,
        should_log=True, should_print=True, name=None):
    """
    Logs content to a specified file and optionally prints it to the console.

    This function handles logging by writing content to a file at the specified 
    file path. If the file does not exist or if the `reset_logs` flag is set to 
    True, it will create a new log file or clear the existing one. The function 
    also offers an option to print the content to the console.

    Parameters
    ----------
    file_path : str or None
        The path to the log file. If None, the function does nothing.
    content : str
        The content to be logged. This will be appended to the file.
    reset_logs : bool
        If True, the log file will be cleared before writing the new content. Default is False.
    reset_output : bool
        If True, the output stream (printed content) gets deleted/cleared.
    should_print : bool
        If True, the content will also be printed to the console. Default is True.
    name : str or None
        Name of the logfile. Else the name of the model.

    Returns
    -------
    None

    Example
    -------
    >>> log("logs/my_log.txt", "This is a log entry.")
    """
    if file_path is None:
        return
    
    # set name
    os.makedirs(file_path, exist_ok=True)
    if not name.endswith(".txt"):
        name += ".txt"
    file_path = os.path.join(file_path, name)

    if should_log:
        if not os.path.exists(file_path) or reset_logs:
            os.makedirs("/".join(file_path.split("/")[:-1]), exist_ok=True)
            with open(file_path, "w") as f:
                f.write("")

        with open(file_path, "a") as f:
            f.write(f"\n{content}")

    if reset_output:
        clear_printing()

    if should_print:
        print(content)

def log_train(epochs, cur_epoch, max_steps, cur_step, n_samples, batch_size, times, losses, log_path, last_time, name):
    """
    Creates informative and helpful logs and prints for training.
    """
    # log/print current epoch and training progress
    output_print = f"\n\n{'#'*24}\n{cur_epoch+1}. Epoch:"
    progress = (cur_epoch+1)/epochs
    blocks = 20
    output_print += f"\n    -> Progress: |{'#'*int(blocks*progress)}{' '*int(blocks*(1-progress))}|"

    log(content=output_print, file_path=log_path, reset_logs=False, reset_output=True, should_log=True, should_print=True, name=name)
    

    # log/print loss and generell information
    cur_time = time.time()
    duration = cur_time - last_time
    times += [duration]
    
    eta_str = str(timedelta(seconds=(max_steps-cur_step) * np.mean(np.array(times)))).split('.')[0]
        
    total_loss = sum([np.mean(np.array(losses[k])) for k in losses.keys()])
    loss_labels = [[key, np.mean(np.array(value))] for key, value in losses.items()]

    update_output(
            cur_epoch=cur_epoch,
            cur_step=cur_step, 
            max_steps=max_steps,
            eta_str=eta_str,
            data_size=n_samples,
            total_loss=total_loss,
            losses=loss_labels,
            batch_size=batch_size,
            log_path=log_path,
            name=name
    )

def update_output(cur_epoch, cur_step, max_steps,
                  data_size, eta_str, total_loss, 
                  losses, batch_size, log_path, name):
    """
    Updates and logs the training output for a model.

    This function formats and prints the current training status, including
    epoch and step details, duration, estimated time of arrival (ETA),
    total loss, and specific loss metrics. It also logs the output to a
    specified log file.

    Parameters
    ----------
    cur_epoch : int
        Current epoch number.
    cur_step : int
        Current step number.
    max_steps : int
        Total number of steps for the training.
    data_size : int
        Total size of the training dataset.
    eta_str : str
        Estimated time of arrival as a string.
    total_loss : float
        Total loss for the current step.
    losses : dict
        Dictionary of specific losses to be displayed.
    batch_size : int
        Size of the batch used in training.
    log_path : str
        Path to the log file where output should be written.

    Returns
    -------
    None
    """
    now = datetime.now()
    output = f"Training - {now.hour:02}:{now.minute:02} {now.day:02}.{now.month:02}.{now.year:04}"

    detail_output = f"\n| epoch: {cur_epoch:>5} || step: {cur_step:>8} || ETA: {eta_str:>8} || total loss: {total_loss:>8.3f} || "
    detail_output += ''.join([f' {key}: {value:>8.3f} |' for key, value in losses])

    steps_in_cur_epoch = cur_step - cur_epoch*(data_size // batch_size)
    cur_epoch_progress =  steps_in_cur_epoch / max(1, data_size // batch_size)
    cur_epoch_progress = min(int((cur_epoch_progress*100)//10), 10)
    cur_epoch_progress_ = max(10-cur_epoch_progress, 0)

    cur_total_progress = cur_step / max_steps
    cur_total_progress = min(int((cur_total_progress*100)//10), 10)
    cur_total_progress_ = max(10-cur_total_progress, 0)

    percentage_output = f"\nTotal Progress: |{'#'*cur_total_progress}{' '*cur_total_progress_}|    Epoch Progress: |{'#'*cur_epoch_progress}{' '*cur_epoch_progress_}|"

    print_output = f"\n\n{'-'*32}\n{output}\n{detail_output}\n{percentage_output}\n"


    # print new output
    clear_printing()

    log(log_path, print_output, name=name)






#############################
######### Classes ###########
#############################

class Layer():
    """
    A class to represent a neural network layer.
    Consists of weights, bias and an optinal activation function.
    """

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
        X : np.array
            Input data for the layer.

        Returns
        -------
        np.array
            Output of the layer after applying the activation function.
        """
        z = np.dot(self.weights, X) + self.bias
        return self.activation_func(z)



class ArtificialNeuralNetwork():
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
        X : np.array
            Input data for the network.

        Returns
        -------
        np.array
            Final output of the network.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X, y, epochs, batch_size=1, 
              parallel_computing=True, shuffle_data=True,
              print_ever_x_steps=10, save_model_every_x_epochs=2,
              log_path="./logs", model_save_path="./models", 
              name=None):
        """
        Trains the artificial neural network (ANN) on the given data.

        Parameters
        ----------
        X : np.array
            Array with multiple datapoints (features).
        y : np.array
            Array with ground truth labels (targets).
        epochs : int
            Amount of runs through the whole data.
        batch_size : int
            Group size of data, forwarding together before adjusting the weights.
        print_ever_x_steps : int
            How often should the informations be logged and printed? Every X steps they will be printed and logged.
        save_model_every_x_epochs : int
            How often should the model be saved? Every X epochs it will be saved.
        log_path : str
            Folder where the logs should be saved.
        model_save_path : str
            Folder where the model should save to.
        name : str or None
            Name of the model / training. Decides the name of the logs and the model saving name. 
            If None, the standard model name will be used.

        Returns
        -------
        ArtificialNeuralNetwork
            Updates ANN Model internally and also returns the model, for connected calls
        """

        # Init
        last_time = time.time()
        times = []
        losses = dict()
        all_losses = []
        n_samples = X.shape[0]
        all_iterations = n_samples//batch_size
        max_steps = epochs * all_iterations
        start_time = get_cur_time_str()
        cur_step = 0

        if not name:
            name = f"{self.name}"

        output_print = f"\n{'-'*24}\n{self.name} Training:\n    - Epochs: {epochs}\n    - Layers: {len(self.layers)}"
        log(content=output_print, file_path=log_path, reset_logs=True, should_log=True, should_print=True, name=name)

        # Go through every datapoint X-times
        for cur_epoch in range(epochs):
            cur_iteration = 0

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
                        loss_dict = self.loss_function(cur_y, y_)    # loss

                        all_losses += [loss_dict]

                        # how to customize prediction_elments
                        prediction_elments = []
                        if self.prediction_elements_tuple["X"]:
                            prediction_elments += [cur_X]
                        if self.prediction_elements_tuple["y"]:
                            prediction_elments += [cur_y]
                        if self.prediction_elements_tuple["y_"]:
                            prediction_elments += [y_]
                        if self.prediction_elements_tuple["error"]:
                            prediction_elments += [loss_dict]
                        prediction_cache += [prediction_elments]

                # adjust weights
                for cur_prediction_element in prediction_cache:
                    self.update_weights(cur_prediction_element)

                # log losses
                for key, value in loss_dict.items():
                    if key in losses.keys():
                        losses[key] += [value]
                    else:
                        losses[key] = [value]
                cur_total_loss = sum([value for value in loss_dict.values()])

                if cur_step > 0 and cur_step % print_ever_x_steps == 0:
                    log_train(epochs, cur_epoch, max_steps, cur_step, n_samples, batch_size, times, losses, log_path, last_time, name)

                    # reset
                    times = []
                    losses = dict()
                    last_time = time.time()
                
                # after every batch
                cur_step += 1

            # add save check
            if cur_step > 0 and cur_step % save_model_every_x_epochs == 0:
                self.save(save_path=model_save_path, name=name+f"_epoch_{cur_epoch}")
        
        # after training
        log_train(epochs, cur_epoch, max_steps, cur_step, n_samples, batch_size, times, losses, log_path, last_time, name)
        end_time = get_cur_time_str()

        self.train_history["start-time"] += [start_time]
        self.train_history["end-time"] += [end_time]
        self.train_history["epochs"] += [cur_epoch]
        self.train_history["total-steps"] += [cur_step]
        self.train_history["batch-size"] += [batch_size]
        self.train_history["all-errors"] += [all_losses]

        save_path = self.save(save_path=model_save_path, name=name)
        log(file_path=log_path, content=f"\n\nCongratulations!!!🥳\n🚀 Your model 🚀 waits here for you:\n      -> '{save_path}'", name=name)

        return self

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
            Adds the new layer internally to the list.
        """
        layer = Layer(input_size, output_size, activation_func)
        self.layers += [layer]

    def save(self, save_path:str, name=None):
        """
        Saves the current neural network to a file using pickle.

        Parameters
        ----------
        save_path : str
            The file path where the model should be saved.

        Returns
        -------
        str
            The path, where the model got saved.
        """
        if name:
            name = name
        else:
            name = f"{self.name}.pkl"

        if not name.endswith(".pkl"):
            name += ".pkl"

        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, name)

        try:
            with open(save_path, 'wb') as file:
                pickle.dump(self, file)
            # print(f"Model saved successfully to {save_path}.")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

        return save_path

    @staticmethod
    def load(save_path: str):
        """
        Loads a neural network from a file using pickle.

        Parameters
        ----------
        save_path : str
            The file path from which the model should be loaded.

        Returns
        -------
        ArtificialNeuralNetwork
            The loaded neural network instance.
        """
        try:
            with open(save_path, 'rb') as file:
                model = pickle.load(file)
            print(f"Model loaded successfully from {save_path}.")
            return model
        except Exception as e:
            raise FileNotFoundError(f"An error occurred while loading the model: {e}")

    def loss_plot(self, width=8, height=5, smoothing_size=1, should_show=True):
        """
        Plots the loss/error of the latest training.

        Parameters
        ----------
        width :  int
            Width of the plot in inch.
        height : int
            Height of the plot in inch.
        smoothing_size : int ( >= 1 and <= total steps)
            Window size for moving average, to smooth the plot.
        should_show : Boolean
            Defines whether to show the plot or not.

        Returns
        -------
        None
            Creates the plot inside and does not return something.
        """
        end_times = self.train_history["end-time"]
        if len(end_times) <= 0:
            return None
        parsed_end_times = [datetime.strptime(time, "%d.%m.%Y %H:%M:%S") for time in end_times]
        latest_index = parsed_end_times.index(max(parsed_end_times))

        loss = self.train_history["all-errors"][latest_index]
        loss = sum([np.mean(np.array(loss[k])) for k in loss.keys()])

        loss = moving_average_sliding_window(loss, window_size=smoothing_size)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))
        ax.plot(np.arange(0, len(all_errors)), all_errors)
        ax.set_xlabel("Time Iterations")
        ax.set_ylabel("Error")
        plt.title("Error over Time")
        # saving?
        if should_show:
            plt.show()

    # UPDATE ME
    def update_weights(self, prediction_element):
        """
        Method to update the weights for convergence.

        Parameters
        ----------
        prediction_element : list
            List of different values, defines by the self.prediction_elements_tuple in the __init__ method.

        Returns
        -------
        None
            Updates the weights internally.
        """
        pass

    # UPDATE ME
    def loss_function(self, y, y_):
        """
        Loss function to calculate the error of the neural network compared to the true value.

        Parameters
        ----------
        y : np.array
            Ground Truth values.
        y_ : np.array
            Predicted values.

        Returns
        -------
        dict
            Loss Values of the prediction.
        """
        return {"Sum Prediction Loss": np.sum(y - y_)}

    # UPDATE ME
    def eval(self, X, y):
        """
        Evaluates the neural network on given testdata, to see the error made on this data.

        Parameters
        ----------
        X : np.array
            Input data.
        y : np.array
            Labeled Ground Truth data.

        Returns
        -------
        float/int
            The absolute error made on the testdata.
        """
        absolute_error = 0

        for i, x in enumerate(X):
            absolute_error += np.abs(y[i] - self.forward(x))

        return absolute_error














