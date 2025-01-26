"""
Loss Functions should be defined as a string where every element is splitted with one space.

You can add more loss-functions here or in your python file/notebook. 
Remember to use nfr to use a numpy function.
With the parameters: function name:str, list parameters, name (dict) parameters, numpy_version:NUMPY_VERSION
"""



###############
### Imports ###
###############

from enum import Enum
# from builtins import sum as sum_, abs as abs_

from .numpy_utils import numpy_function_retriever as nfr, NUMPY_VERSION



#############
### Tools ###
#############

def get_total_loss(loss_dict):
    """
    Changes a dict of losses to a total loss.
    """
    return sum([value for value in loss_dict.values()])



######################
### Loss Functions ###
######################

def sum_error(y, y_, numpy_version:NUMPY_VERSION):
    return nfr("sum", list_params=[y - y_], numpy_version=numpy_version)

# FIXME: also change the other examples to nfr

def sum_absolute_error(y, y_, numpy_version:NUMPY_VERSION):
    return sum( abs(y - y_, numpy_version=numpy_version), numpy_version=numpy_version )

def mean_absolute_error(y, y_, numpy_version:NUMPY_VERSION):
    return mean( abs(y - y_, numpy_version=numpy_version), numpy_version=numpy_version )

def mean_squared_error(y, y_, numpy_version:NUMPY_VERSION):
    return mean( (y - y_)**2, numpy_version=numpy_version )

def mean_root_squared_error(y, y_, numpy_version:NUMPY_VERSION):
    return sqrt( mean_squared_error(y, y_, numpy_version=numpy_version), numpy_version=numpy_version )

def huber_loss(y, y_, numpy_version:NUMPY_VERSION):
    delta = 1.0
    residuals = y - y_
    huber_loss = where(
        abs(residuals, numpy_version=numpy_version) <= delta,
        0.5 * residuals**2,
        delta * abs(residuals, numpy_version=numpy_version) - 0.5 * delta**2,
        numpy_version=numpy_version
    )
    return mean(huber_loss, numpy_version=numpy_version)

def cost_function(y, y_, numpy_version:NUMPY_VERSION):
    return 0.5 * power(y_ - y, 2, numpy_version=numpy_version)

# ADD YOUR CUSTOM LOSS-FUNCTIONS HERE OR IN YOUR PYTHON FILE / NOTEBOOK
# ...





