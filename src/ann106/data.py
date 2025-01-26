"""
This file is handling data, preparing it, importing it and generating it.
"""

import pickle
from enum import Enum
from typing import Union, List, Any

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from .numpy_utils import NUMPY_VERSION


class COMPUTING_DEVICE(Enum):
    CPU = np
    GPU = cp
    # ...?

# also different GPU methods? Only CuPy?

class DataTensor():
    def __init__(self, computing_device:COMPUTING_DEVICE=COMPUTING_DEVICE.CPU):
        """
        Initialize the DataTensor class. Starts with an empty tensor.

        :param computing_device: The device to use (CPU or GPU).
        :type computing_device: COMPUTING_DEVICE
        """
        self.device = computing_device
        self.history = []  # Stores history of changes or operations
        self.data = None   # Stores the data (numpy or cupy array)

    def add_history(self, operation:str, *list_args, **dict_args):
        """
        Adds an operation performed on the data to the history.
        """
        self.history.append({"operation":operation, "args":list_args, "named_args":dict_args})

    def get(self):
        """
        Returns a copy of the data (numpy or cupy array).
        """
        return self.data.copy()

    def send_to(self, target_device: COMPUTING_DEVICE):
        """
        Send data to the specified computing device (CPU or GPU or ...?).
        """
        if self.device == target_device:
            return

        if target_device == COMPUTING_DEVICE.CPU:
            self.data = cp.asnumpy(self.data)
        elif target_device == COMPUTING_DEVICE.GPU:
            self.data = cp.asarray(self.data)
        else:
            raise ValueError(f"Unknown target device: {target_device}")

        self.device = target_device
        self.save_history("send_to", target_device)

        return self

    def import_data(self, source: Union[np.ndarray, cp.ndarray]):
        """
        Imports data from a existing numpy or cupy array.
        """
        if isinstance(source, np.ndarray):
            if self.device == COMPUTING_DEVICE.GPU:
                self.data = cp.asarray(source)
            else:
                self.data = source
        elif isinstance(source, cp.ndarray):
            if self.device == COMPUTING_DEVICE.CPU:
                self.data = cp.asnumpy(source)
            else:
                self.data = source
        else:
            raise TypeError("Source must be a numpy or cupy array.")

        self.save_history("import_data", type(source))

        return self

    def get_numpy_version(self):
        if isinstance(self.data, np.ndarray):
            return NUMPY_VERSION.NUMPY
        elif isinstance(self.data, cp.ndarray):
            return NUMPY_VERSION.CUPY
        else:
            raise ValueError("Your data is empty! Can't extract the numpy version from empty data.")

    # def apply_numpy_function(self, func_name:str, list_params:list=None, dict_params:dict=None):
    #     numpy_lib = self.device.value
    #     try:
    #         func = getattr(numpy_lib, func_name)  # Dynamically get the function
    #     except AttributeError:
    #         raise ValueError(f"The function '{func_name}' does not exist in {numpy_lib.__name__}.")

    #     # Call the function with the provided list and dict parameters
    #     if list_params and dict_params:
    #         return func(*list_params, **dict_params)
    #     elif list_params and not dict_params:
    #         return func(*list_params)
    #     elif not list_params and dict_params:
    #         return func(*dict_params)
    #     elif not list_params and not dict_params:
    #         return func()

    # adding saving and loading of Data objects?

    # data saving
    def save_numpy_array(self, file_path: str):
        """Save the current data to a file."""
        if self.data is None:
            raise ValueError("No data to save.")

        if self.device == COMPUTING_DEVICE.GPU:
            np.save(file_path, cp.asnumpy(self.data))
        else:
            np.save(file_path, self.data)
        self.save_history("save_to_file", file_path)
    # add more saving options

    # data loading -> images, data tables, text, audio, ..?
    def load_numpy_array(self, file_path: str):
        """
        Loads a numpy array as data from a file.
        """
        self.data = np.load(file_path)
        if self.device == COMPUTING_DEVICE.GPU:
            self.data = cp.asarray(self.data)
        self.save_history("load_from_file", file_path)

        return self

    # add more loading options!

    def clear_data(self):
        """
        Clear the current data obj and reset history.
        """
        self.data = None
        self.add_history("clear_data")

    # data generation functions -> n dim, float? -> for different tasks?
    def generate_data(self, shape: tuple, value: float = 0.0):
        """Generate data with a specified shape and fill it with a value."""
        if self.device == COMPUTING_DEVICE.CPU:
            self.data = np.full(shape, value, dtype=np.float32)
        elif self.device == COMPUTING_DEVICE.GPU:
            self.data = cp.full(shape, value, dtype=cp.float32)
        self.save_history("generate_data", shape, value)

        return self
    
    # ...

    # data checks
    def check_data(self) -> str:
        """
        Check and return the status of the current data.

        -> add more information?
        """
        if self.data is None:
            return "No data loaded."
        return f"Data loaded on {self.device.value} with shape {self.data.shape}"


# def gradient_func(func, x_values):
#     return np.array([derivative(func, x, dx=1e-5) for x in x_values]).reshape(-1, 1)


# Integrate to the data class?
def data_generator_binary_classification(
                            x_value_range_class_1=(-8, 8), y_value_range_class_1=(1, 5), 
                            x_value_range_class_2=(-8, 8), y_value_range_class_2=(-5, -1), 
                            value_amount_class_1=200, value_amount_class_2=150,
                            func=np.sin, plotting=True, plot_style=None):
    """
    This function can be used to generate 2 dimensional data for binary classification.
    It uses the functions to generate the datapoints. 
    The generated points have an orthogonal offset towards the function.
    
    :param x_value_range_class_1: Sets the ranges of X values for the class 0.
    :type x_value_range_class_1: tuple(float, float)
    :param y_value_range_class_1: Sets the ranges of the y offset values for the class 0. The y values will be generated with the x values and the function func with this additional offset.
    :type y_value_range_class_1: tuple(float, float)
    :param x_value_range_class_2: Sets the ranges of X values for the class 1.
    :type x_value_range_class_2: tuple(float, float)
    :param y_value_range_class_2: Sets the ranges of the y offset values for the class 1. The y values will be generated with the x values and the function func with this additional offset.
    :type y_value_range_class_2: tuple(float, float)
    :param value_amount_class_1: Sets the amount of datapoints for the class 0.
    :type value_amount_class_1: int
    :param value_amount_class_2: Sets the amount of datapoints for the class 1.
    :type value_amount_class_2: int
    :param func: Function which will be used to generate the y coordinate values.
    :type func: callable
    :param plotting: Decides whether to plot the generated datapoints or not.
    :type plotting: bool
    :param plot_style: Defines the style of the plot. If None, the style will be choosen randomly.
    :type plot_style: str or None

    :return: A tuple containing the generated data -> as X and y.
    :rtype: tuple(np.array, np.array)

    :example:

    >>> import ann106 as ann
    >>> import numpy as np  
    >>> X, y = ann.data.generate_binary_classification_data(
    ...     x_value_range_class_1=(-8, 8),
    ...     y_value_range_class_1=(0.5, 0.5),
    ...     x_value_range_class_2=(-8, 8),
    ...     y_value_range_class_2=(0.5, 0.5),
    ...     value_amount_class_1=500,
    ...     value_amount_class_2=150,
    ...     func=np.sin,
    ...     plotting=True,
    ...     plot_style=None
    ... )
    """

    # First get random X-Axis values
    # Then get the y value with the given function and apply a offset

    # Generate X values -> Coordinates X_1, Y_1 and X_2, Y_2
    X_1 = np.random.uniform(x_value_range_class_1[0], x_value_range_class_1[1], value_amount_class_1).reshape(-1, 1)
    Y_1 = func(X_1) # + np.random.uniform(y_value_range_class_1[0], y_value_range_class_1[1], size=X_1.shape)
    X_2 = np.random.uniform(x_value_range_class_2[0], x_value_range_class_2[1], value_amount_class_2).reshape(-1, 1)
    Y_2 = func(X_2) # + np.random.uniform(y_value_range_class_2[0], y_value_range_class_2[1], size=X_2.shape)
    
    # Compute gradients -> slope of the func at the x-values
    epsilon = 1e-5
    # The gradient is calculated numerically for each x-value.
    # This is done using a finite difference approximation, where epsilon is a small step size.
    gradient_1 = (func(X_1+epsilon) - func(X_1)) / epsilon
    gradient_2 = (func(X_2+epsilon) - func(X_2)) / epsilon
    
    # Calculate the orthogonal direction to the gradient.
    # The direction vector (dx_1, dy_1) is normalized to ensure it has unit length.
    dx_1 = -gradient_1 / np.sqrt(1 + gradient_1**2)
    dy_1 = 1 / np.sqrt(1 + gradient_1**2)
    dx_2 = -gradient_2 / np.sqrt(1 + gradient_2**2)
    dy_2 = 1 / np.sqrt(1 + gradient_2**2)

    # Apply random orthogonal offset to the gradient
    offset_1 = np.random.uniform(y_value_range_class_1[0], y_value_range_class_1[1], size=X_1.shape)
    X_1_offset = X_1 + offset_1 * dx_1
    Y_1_offset = Y_1 + offset_1 * dy_1
    offset_2 = np.random.uniform(y_value_range_class_2[0], y_value_range_class_2[1], size=X_2.shape)
    X_2_offset = X_2 + offset_2 * dx_2
    Y_2_offset = Y_2 + offset_2 * dy_2

    XY_1 = np.concat((X_1_offset, Y_1_offset), axis=1)
    XY_2 = np.concat((X_2_offset, Y_2_offset), axis=1)
    
    X = np.concat((XY_1, XY_2), axis=0)

    y = np.concat((np.full((value_amount_class_1), 0), np.full((value_amount_class_2), 1)), axis=0)
    
    if plotting:
        colors = ["#ff7f0e" if cur_y == 1 else "#1f77b4" for cur_y in y]

        if plot_style:
            style_name = plot_style
        else:
            style_name = np.random.choice(plt.style.available)
        print(f"Matplotlib Style: {style_name}")
        plt.style.use(style_name)  # Other options: 'ggplot', 'fivethirtyeight', 'bmh', 'dark_background'
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        ax.scatter(X[:,0], X[:,1], c=colors, s=50, edgecolor='k', alpha=0.8)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title("Data Generation Plot", fontsize=14, fontweight='bold')
        ax.set_xlabel("X Axis", fontsize=12)
        ax.set_ylabel("Y Axis", fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.show()

        # reset plot style
        plt.style.use("default")

    return X, y


