"""
Loss Functions should be defined as a string where every element is splitted with one space.

You can add more loss-functions here or in your python file/notebook. 
Remember to use nfr to use a numpy function.
With the parameters: function name:str, list parameters, name (dict) parameters, numpy_version:NUMPY_VERSION
"""



###############
### Imports ###
###############

from typing import Union, List, Any
from enum import Enum
# from builtins import sum as sum_, abs as abs_

import numpy as np

from autograd import grad as auto_grad
import autograd.numpy as anp

from jax import grad as jax_grad
import jax.numpy as jnp

from .numpy_utils import numpy_function_retriever as nfr, NUMPY_VERSION
from .data import DataTensor



###################
### Definitions ###
###################
class GRADIENT_LIB(Enum):
    AUTOGRAD = anp
    JAX = jnp



#############
### Tools ###
#############

def get_total_loss(loss_dict):
    """
    Changes a dict of losses to a total loss.
    """
    return sum([value for value in loss_dict.values()])



def get_negative_gradient(x:np.ndarray, loss_function, used_lib:GRADIENT_LIB):
    """
    Calculates the negative gradient on a given point.
    The direction to the lowest point of the given loss function.

    :param x: Input Data for finding negative gradient
    :type x: Union(np.ndarray, DataTensor)

    FIXME ...
    """

    if isinstance(x, DataTensor):
        x = x.get()

    # Transform in numpy version arrays of input array?
    # ...

    if used_lib == GRADIENT_LIB.AUTOGRAD:
        # Calculate the derivative and greatest slope
        gradient = auto_grad(loss_function)
    elif used_lib == GRADIENT_LIB.JAX:
        # Calculate the derivative and greatest slope
        gradient = jax_grad(loss_function)
    else:
        raise ValueError(f"Can't use lib '{used_lib}' for calculate the gradient.")

    # Calculate the negative gradient on the given point
    negative_gradient = -gradient(x)


######################
### Loss Functions ###
######################

def sum_error(y, y_, numpy_version:NUMPY_VERSION):
    return nfr("sum", list_params=[y - y_], numpy_version=numpy_version)

def sum_absolute_error(y, y_, numpy_version:NUMPY_VERSION):
    # return sum( abs(y - y_, numpy_version=numpy_version), numpy_version=numpy_version )
    return nfr("sum", list_params=[nfr("abs", list_params=[y - y_], numpy_version=numpy_version)], numpy_version=numpy_version )

def mean_absolute_error(y, y_, numpy_version:NUMPY_VERSION):
    # return mean( abs(y - y_, numpy_version=numpy_version), numpy_version=numpy_version )
    return nfr("mean", list_params=[nfr("abs", list_params=[y - y_], numpy_version=numpy_version)], numpy_version=numpy_version )

def mean_squared_error(y, y_, numpy_version:NUMPY_VERSION):
    # return mean( (y - y_)**2, numpy_version=numpy_version )
    return nfr("mean", list_params=[(y - y_)**2], numpy_version=numpy_version)

def mean_root_squared_error(y, y_, numpy_version:NUMPY_VERSION):
    return nfr("sqrt", list_params=[mean_squared_error(y, y_, numpy_version=numpy_version)], numpy_version=numpy_version )

def huber_loss(y, y_, numpy_version:NUMPY_VERSION):
    delta = 1.0
    residuals = y - y_
    huber_loss = nfr("where",
        list_params=[
            nfr("abs", list_params=[residuals], numpy_version=numpy_version) <= delta,
            0.5 * residuals**2,
            delta * nfr("abs", list_params=[residuals], numpy_version=numpy_version) - 0.5 * delta**2
        ],
        numpy_version=numpy_version
    )
    return nfr("mean", list_params=[huber_loss], numpy_version=numpy_version)

def cost_function(y, y_, numpy_version:NUMPY_VERSION):
    return 0.5 * nfr("power", list_params=[y_ - y, 2], numpy_version=numpy_version)

# ADD YOUR CUSTOM LOSS-FUNCTIONS HERE OR IN YOUR PYTHON FILE / NOTEBOOK
# ...





