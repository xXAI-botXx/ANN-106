"""
A simple Framework for artificial neural networks. With prints, logs and more.
It is recommended to use PyTorch or TensorFlow for professional projects, this is just an educational and fun project.


:features:

- Weight representation through layers (Layer class)
- Customizable ANN
- Multiple Loss functions
- Sklearn Version
- Train History (Time, Errors)
- Lossplotting
- Added Batch-Size and general training loop
- Plotting (as string and as graph)


:planned-features:

- Add backward -> learning
- Add parallel processing (of batches)
    - Parallel(n_jobs=-1)(delayed(compute_square)(num) for num in numbers
- Add more Document Strings -> see: https://github.com/xXAI-botXx/Project-Helper/blob/main/guides/Sphinx_Helper.md#How-you-should-code
- Add GPU calculation
- Add validation data during training
- Add Dataloader (?)
- Add more networks -> CNN, ...


:example:

>>> class MyANN(ann.base.ArtificialNeuralNetwork):
...    def __init__(self):
...        super().__init__()
...        self.prediction_elements_tuple = {
...            "X": True,
...            "y": True,
...            "y_": False,
...            "all_y_": True,
...            "error": False
...        }
...        self.name = "MyANN"
...        self.layers = [
...                ann.base.Layer(2, 1, None), 
...                    ]

>>>    def update_weights(self, prediction_element):
...        # extract needed elements
...        cur_X, cur_y, cur_y_pred = prediction_element 
...        cur_y_pred = cur_y_pred[0]
...        cur_error = cur_y - cur_y_pred
        
...        delta_weights = self.get_lr() * cur_error * cur_X
...        self.layers[0].weights = self.layers[0].weights+delta_weights

...        self.layers[0].bias = self.layers[0].bias + self.get_lr() * cur_error

>>>    def loss_function(self, y, y_):
...        return {"Sum Loss":ann.loss_functions.sum_error(y, y_)} # y - y_

>>>    def predict(self, x):
...        x = self.scale(x)
...        # return self.forward(x)
...        return ann.activation_functions.step_function(self.forward(x), threshold=0.0, greater_equal_value=1, smaller_value=-1)

>>> model = MyANN()
>>> learn_rate_scheduler = ann.learn_rate.LearnrateScheduler(start_learnrate=0.005)
>>> model.train(X=X, y=y, epochs=500, parallel_computing=False, print_ever_x_steps=1000, learn_rate_scheduler=learn_rate_scheduler)
>>> model.eval(X=X, y=y)



Author: Tobia Ippolito
"""

#############################
########## Imports ##########
#############################

# from copy import deepcopy
import sys
import os
import pickle
import inspect
from copy import deepcopy
from datetime import datetime, timedelta
import time

from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from .learn_rate import LearnrateScheduler
from .activation_functions import pass_through, step_function
from .loss_functions import sum_absolute_error





#############################
##### Helper Functions ######
#############################

def get_cur_time_str():
    """
    Returns the current time and date as string: day.month.year hour:minute:second
    """
    now = datetime.now()
    return f"{now.day}.{now.month}.{now.year} {now.hour}:{now.minute}:{now.second}"

def moving_average_sliding_window(input_data:list, window_size):
    """
    Smooths a list moving a window over the data and applying the average on the window.
    """
    data = np.array(input_data)
    if data.shape == None:
        data = np.array([input_data])
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

    :param file_path: The path to the log file. If None, the function does nothing.
    :type file_path: str or None
    :param content: The content to be logged. This will be appended to the file.
    :type content: str
    :param reset_logs: If True, the log file will be cleared before writing the new content. Default is False.
    :type reset_logs:  bool
        
    :param reset_output: If True, the output stream (printed content) gets deleted/cleared.
    :type reset_output: bool
    :param should_print: If True, the content will also be printed to the console. Default is True.
    :type should_print: bool
    :param name: Name of the logfile. Else the name of the model.
    :type name: str or None
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

    :param cur_epoch: Current epoch number.
    :type cur_epoch: int
    :param cur_step: Current step number.
    :type cur_step: int
    :param max_steps: Total number of steps for the training.s
    :type max_steps: int
    :param data_size: Total size of the training dataset.
    :type data_size: int
    :param eta_str: Estimated time of arrival as a string.
    :type eta_str: str
    :param total_loss: Total loss for the current step.
    :type total_loss: float
    :param losses: Dictionary of specific losses to be displayed.
    :type losses: dict
    :param batch_size: Size of the batch used in training.
    :type batch_size: int
    :param log_path: Path to the log file where output should be written.
    :type log_path: str
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

    clear_printing()
    
    # print new output
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

        :param input_size: Number of input neurons to the layer.
        :type input_size: int
        :param output_size: Number of output neurons from the layer.
        :type output_size: int
        :param activation_func: Activation function for the layer.
        :type activation_func: callable or None
        """
        self.weights = np.random.normal(0, 0.01, (output_size, input_size)) if output_size is not None and input_size is not None else None
        self.bias = np.random.normal(0, 0.01, (output_size,)) if output_size is not None else None
        self.activation_func = activation_func if activation_func is not None else pass_through # lambda x: x 

    def forward(self, X):
        """
        Forward pass through the layer.

        :param X: Input data for the layer.
        :type X: np.array

        :return: Output of the layer after applying the activation function.
        :rtype: np.array
        """
        if self.weights is not None:
            z = np.dot(self.weights, X)
        else:
            z = X

        if self.bias is not None:
            z += self.bias

        return self.activation_func(z)



class ArtificialNeuralNetwork():
    def __init__(self, learn_rate_scheduler=None):
        self.layers = []
        self.name = "ANN"
        self.learn_rate_scheduler = learn_rate_scheduler if learn_rate_scheduler is not None else LearnrateScheduler(start_learnrate=1.0)

        # choose which elments you want for your "update_weights" method
        self.prediction_elements_tuple = {
            "X": True,
            "y": False,
            "y_": False,
            "all_y_": False,
            "error": True
        }

        self.train_history = {
            "start-time":[],
            "end-time":[],
            "epochs":[],
            "total-steps":[],
            "batch-size":[],
            "all-errors":[],
            "lr-scheduler":[],
            "end-steps":[],
            "end-epochs":[]
        }
        
    def forward(self, X, return_all_layer_outputs=False):
        """
        Forward pass through the entire network.

        :param X: Input data for the network.
        :type X: np.array

        :return: Final output of the network.
        :rtype: np.array
        """
        all_outputs = []
        output = X
        for layer in self.layers:
            output = layer.forward(output)
            all_outputs += [output]
        
        if return_all_layer_outputs:
            return all_outputs
        else:
            if np.isnan(output) or np.isinf(output):
                print("Warning! Nan or Inf values detected. This may be caused through a too high learning rate.")
            return output

    def train(self, X, y, epochs, batch_size=1, 
              parallel_computing=True, shuffle_data=True,
              print_ever_x_steps=10, save_model_every_x_epochs=2,
              log_path="./logs", model_save_path="./models", 
              name=None, learn_rate_scheduler=None, scaling=True):
        """
        Trains the artificial neural network (ANN) on the given data.

        :param X: Array with multiple datapoints (features).
        :type X: np.array
        :param y: Array with ground truth labels (targets).
        :type y: np.array
        :param epochs: Amount of runs through the whole data.
        :type epochs: int
        :param batch_size: Group size of data, forwarding together before adjusting the weights.
        :type batch_size: int
        :param print_ever_x_steps: How often should the informations be logged and printed? Every X steps they will be printed and logged.
        :type print_ever_x_steps: int
        :param save_model_every_x_epochs: How often should the model be saved? Every X epochs it will be saved.
        :type save_model_every_x_epochs: int
        :param log_path: Folder where the logs should be saved.
        :type log_path: str
        :param model_save_path: Folder where the model should save to.
        :type model_save_path: str
        :param name: Name of the model / training. Decides the name of the logs and the model saving name. If None, the standard model name will be used.
        :type name: str
        :param learn_rate_scheduler: Defines the degree of adjustment of the weights and the strategy to change this adjustment.
        :type learn_rate_scheduler: LearnrateScheduler
        :param scaling: Decides whether to scale input data or not
        :type scaling: bool 

        :return: Updates ANN Model internally and also returns the model, for connected calls. 
        :rtype: ArtificialNeuralNetwork 
        """
        if learn_rate_scheduler:
            self.learn_rate_scheduler = learn_rate_scheduler
        
        # scale data
        if scaling:
            try:
                if self.scaler is None:
                    self.create_scaler(X)
            except Exception:
                self.create_scaler(X)

            X = self.scaler.transform(X)
        else:
            self.scaler = None

        # Init
        last_time = time.time()
        times = []
        losses = dict()
        all_losses = dict()
        n_samples = X.shape[0]
        all_iterations = n_samples//batch_size
        max_steps = epochs * all_iterations
        start_time = get_cur_time_str()
        cur_step = 0

        if not name:
            name = f"{self.name}"

        output_print = f"\n{'-'*24}\n{self.name} Training:\n    - Epochs: {epochs}\n    - Layers: {len(self.layers)}"
        log(content=output_print, file_path=log_path, reset_logs=True, should_log=True, should_print=True, name=name)

        try:
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
                            if self.prediction_elements_tuple["all_y_"]:
                                all_y_ = self.forward(cur_X, return_all_layer_outputs=True)
                                y_ = all_y_[-1]
                            else:
                                y_ = self.forward(cur_X) 
                            loss_dict = self.loss_function(cur_y, y_)    # loss

                            for cur_loss_name, cur_loss_value in loss_dict.items():
                                if cur_loss_name in all_losses.keys():
                                    all_losses[cur_loss_name] += [float(cur_loss_value)]
                                else:
                                    all_losses[cur_loss_name] = [float(cur_loss_value)]

                            # how to customize prediction_elments
                            prediction_elments = []
                            if self.prediction_elements_tuple["X"]:
                                prediction_elments += [cur_X]
                            if self.prediction_elements_tuple["y"]:
                                prediction_elments += [cur_y]
                            if self.prediction_elements_tuple["y_"]:
                                prediction_elments += [y_]
                            if self.prediction_elements_tuple["all_y_"]:
                                prediction_elments += [all_y_]
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
                    self.learn_rate_scheduler.step()

                # add save check
                if cur_step > 0 and cur_step % save_model_every_x_epochs == 0:
                    self.save(save_path=model_save_path, name=name+f"_epoch_{cur_epoch}")
        except KeyboardInterrupt:
            log(content="Early Stop Initiated!", file_path=log_path, reset_logs=False, reset_output=True, should_log=True, should_print=True, name=name)

            log_train(epochs, cur_epoch, max_steps, cur_step, n_samples, batch_size, times, losses, log_path, last_time, name)
            end_time = get_cur_time_str()

            self.train_history["start-time"] += [start_time]
            self.train_history["end-time"] += [end_time]
            self.train_history["epochs"] += [epochs]
            self.train_history["total-steps"] += [max_steps]
            self.train_history["batch-size"] += [batch_size]
            self.train_history["all-errors"] += [all_losses]
            self.train_history["lr-scheduler"] += [self.learn_rate_scheduler]
            self.train_history["end-steps"] += [cur_step]
            self.train_history["end-epochs"] += [cur_epoch]

            save_path = self.save(save_path=model_save_path, name=name)
            log(file_path=log_path, content=f"\n\n🚀 Your model 🚀 waits here for you:\n      -> '{save_path}'", name=name)

        

        # after training
        log_train(epochs, cur_epoch, max_steps, cur_step, n_samples, batch_size, times, losses, log_path, last_time, name)
        end_time = get_cur_time_str()

        self.train_history["start-time"] += [start_time]
        self.train_history["end-time"] += [end_time]
        self.train_history["epochs"] += [epochs]
        self.train_history["total-steps"] += [max_steps]
        self.train_history["batch-size"] += [batch_size]
        self.train_history["all-errors"] += [all_losses]
        self.train_history["lr-scheduler"] += [self.learn_rate_scheduler]
        self.train_history["end-steps"] += [cur_step]
        self.train_history["end-epochs"] += [cur_epoch]

        save_path = self.save(save_path=model_save_path, name=name)
        log(file_path=log_path, content=f"\n\nCongratulations!!!🥳\n🚀 Your model 🚀 waits here for you:\n      -> '{save_path}'", name=name)

        return self

    def get_lr(self):
        try:
            return self.learn_rate_scheduler.learn_rate
        except Exception:
            return None

    def add_layer(self, input_size, output_size, activation_func=None):
        """
        Adds a new layer to the neural network.

        :param input_size: Number of input neurons to the layer.
        :type input_size: int
        :param output_size: Number of output neurons from the layer.
        :type output_size: int
        :param activation_func: Activation function for the layer.
        :type activation_func: callable or None
        """
        layer = Layer(input_size, output_size, activation_func)
        self.layers += [layer]

    def save(self, save_path:str, name=None):
        """
        Saves the current neural network to a file using pickle.

        :param save_path: The file path where the model should be saved.
        :type save_path: str
        :param name: Name of the model.
        :type name: str or None

        :return: The path, where the model got saved.
        :rtype: str
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

        :param save_path: The file path from which the model should be loaded.
        :type save_path: str
            
        :return: The loaded neural network instance.
        :rtype: ArtificialNeuralNetwork
        """
        try:
            with open(save_path, 'rb') as file:
                model = pickle.load(file)
            print(f"Model loaded successfully from {save_path}.")
            return model
        except Exception as e:
            raise FileNotFoundError(f"An error occurred while loading the model: {e}")

    def loss_plot(self, width=8, height=5, smoothing_size=1, should_show=True, loss_name=None):
        """
        Plots the loss/error of the latest training.

        :param width: Width of the plot in inch.
        :type width: int
        :param height: Height of the plot in inch.
        :type height: int
        :param smoothing_size: Window size for moving average, to smooth the plot. 
        :type smoothing_size: int ( >= 1 and <= total steps)
        :param should_show: Defines whether to show the plot or not.
        :type should_show: bool
        :param loss_name: To get the loss plot for one specific loss. If None, the losses get summed/stacked.
        :type loss_name: str
        """
        end_times = self.train_history["end-time"]
        if len(end_times) <= 0:
            return None
        parsed_end_times = [datetime.strptime(time, "%d.%m.%Y %H:%M:%S") for time in end_times]
        latest_index = parsed_end_times.index(max(parsed_end_times))

        loss = self.train_history["all-errors"][latest_index]
        for key, values in loss.items():
            indexes = len(values)
            break
        total_loss = [0]*indexes
        for key, values in loss.items():
            if loss_name is not None:
                if key != loss_name:
                    continue
            for i, value in enumerate(values):
                total_loss[i] += value

        if smoothing_size > 1:
            total_loss = moving_average_sliding_window(total_loss, window_size=smoothing_size)

        plot_name = "Total Loss"
        if loss_name is not None:
            plot_name = loss_name

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))
        ax.plot(np.arange(0, len(total_loss)), total_loss)
        ax.set_xlabel("Time Iterations")
        ax.set_ylabel("Loss")
        plt.title(f"{plot_name} over Time")
        # saving?
        if should_show:
            plt.show()

    def __str__(self):
        """
        Returns the architecture of the neural network
        as well as the Output Function, the Loss Function and the Update Weights Function.
        
        :return: The architecture as string.
        :rtype: str

        :example:

        ################################################################

        MLP - Architecture:
        --------------------------------
        Input: 2, Output: 2, Activation: pass_through
        --------------------------------
        Input: 2, Output: 1, Activation: pass_through
        --------------------------------
        Output Function:
            def predict(self, x):
                x = self.scale(x)
                # return self.forward(x)
                return ann.activation_functions.step_function(self.forward(x), threshold=0.0, greater_equal_value=1, smaller_value=-1)


        Loss Function:
            def loss_function(self, y, y_):
                return {"Sum Loss":ann.loss_functions.sum_error(y, y_)} # y - y_


        Update Weights Function:
            def update_weights(self, prediction_element):
                # extract needed elements
                cur_X, cur_y, cur_y_pred = prediction_element 
                cur_y_pred = cur_y_pred[0]
                cur_error = cur_y - cur_y_pred
                
                delta_weights = self.get_lr() * cur_error * cur_X
                self.layers[0].weights = self.layers[0].weights+delta_weights

                self.layers[0].bias = self.layers[0].bias + self.get_lr() * cur_error


        ################################################################

        """
        ann_str = f"\n{'#'*64}\n\n"
        ann_str += f"{self.name} - Architecture:"
        ann_str += self.get_architecture()
        ann_str += f"\n{'-'*32}\nOutput Function:\n{inspect.getsource(self.predict)}"

        ann_str += f"\n\nLoss Function:\n{inspect.getsource(self.loss_function)}"

        ann_str += f"\n\nUpdate Weights Function:\n{inspect.getsource(self.update_weights)}"

        ann_str += f"\n\n{'#'*64}"
        return ann_str

    def get_architecture(self):
        """
        Returns only the layers/architecture of the ANN.

        :return: Layers of ANN.
        :rtype: str
        """
        ann_str = ""
        for layer in self.layers:
            ann_str += f"\n{'-'*32}\nInput: {layer.weights.shape[1]}, Output: {layer.weights.shape[0]}, Activation: {layer.activation_func.__name__}"
        return ann_str

    def to_graph(self):
        """
        Transforms the Neural Network into an Graph.

        :return: The nodes, edges and the position of every node.
        :rtype: tuple(list(int), list(tuple(int, int)), dict(int: tuple(int, int)))
        """
        nodes = []
        edges = []
        pos = []
        index = 0
        last_nodes = None
        for layer in self.layers:
            if len(nodes) == 0:
                nodes += np.arange(0, layer.weights.shape[1]).tolist()
                index = layer.weights.shape[1]
                last_nodes = deepcopy(nodes)
                pos += list(zip(np.zeros(layer.weights.shape[1]).tolist(), np.arange(0, layer.weights.shape[1]).tolist()))

            new_layer = np.arange(index, layer.weights.shape[0]+index).tolist()
            nodes += new_layer

            # print("\n\nNew Edges:\n")
            # print("Index:", index, " last_nodes:", last_nodes, " cur_nodes:", new_layer, "\n")
            for cur_last_node in last_nodes:
                for cur_node in new_layer:
                    # print(cur_last_node, cur_node)
                    edges += [(cur_last_node, cur_node)]

            pos += list(zip((np.ones(layer.weights.shape[0])*index).tolist(), np.arange(0, layer.weights.shape[0]).tolist()))
            
            index = new_layer[-1]+1
            last_nodes = deepcopy(new_layer)

        # Adjust positions -> transform to dict
        pos_dict = dict()
        for i, cur_node in enumerate(nodes):
            pos_dict[cur_node] = pos[i]

        return nodes, edges, pos_dict

    # works currently only for fully connected
    def plot_architecture(self):
        # Get neurons/nodes and weights/edges
        nodes, edges, pos_dict = self.to_graph()

        # Create directed Graph with nodes and edges
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        # Plot graph
        nx.draw(graph, pos_dict, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
        plt.show()

    def create_scaler(self, X):
        scaler = MinMaxScaler()
        scaler = scaler.fit(X)
        self.scaler = scaler
        self.min_scaler_value = np.min(X)
        self.max_scaler_value = np.max(X)

    def scale(self, x):
        try:
            if self.scaler is None:
                return x

            x = self.scaler.transform(x.reshape(1, -1))
            return np.squeeze(x)
        except Exception as e:
            # print(e)
            return x

    def eval(self, X, y, loss_func=sum_absolute_error, group_function=sum):
        """
        Evaluates the neural network on given testdata, to see the error made on this data.

        :param X: Input data.
        :type X: np.array
        :param y: Labeled Ground Truth data.
        :type y: np.array
        :param loss_func: Loss metric to calc the error.
        :type loss_func: callable
        :param group_function: Decides how the single error values get put together. IF give a None, ithis method return an list with individual losses.
        :type group_function: callable or None

        :return: The absolute error made on the testdata.
        :rtype: float/int
        """
        error = []

        for i, x in enumerate(X):
            error += [loss_func(y[i], self.predict(x))]

        if group_function is not None:
            return group_function(error)
        else:
            return error

    # UPDATE ME
    def update_weights(self, prediction_element):
        """
        Method to update the weights for convergence.

        :param prediction_element: List of different values, defines by the self.prediction_elements_tuple in the __init__ method.
        :type prediction_element: list
        """
        pass

    # UPDATE ME
    def loss_function(self, y, y_):
        """
        Loss function to calculate the error of the neural network compared to the true value.

        :param y: Ground Truth values.
        :type y: np.array
        :param y_: Predicted values.
        :type y_: np.array

        :return: Loss Values of the prediction.
        :rtype: dict
        """
        return {"Sum Prediction Loss": np.sum(y - y_)}

    # UPDATE ME
    def predict(self, x):
        """
        Method to inference an input. It makes the same as forward 
        but most likely applies a function to come from likelihoods to
        a clean result -> Output Function.

        :param prediction_element: List of different values, defines by the self.prediction_elements_tuple in the __init__ method.
        :type prediction_element: list
        """
        x = self.scale(x)
        return self.forward(x)














