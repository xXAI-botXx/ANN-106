"""
This file is handling data, preparing it, importing it and generating it.
"""

import numpy as np
import matplotlib.pyplot as plt


def gradient_func(func, x_values):
    return np.array([derivative(func, x, dx=1e-5) for x in x_values]).reshape(-1, 1)


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


