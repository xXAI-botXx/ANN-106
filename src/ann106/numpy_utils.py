"""
This file can be used to get numpy functions from different numpy versions.
"""

###############
### Imports ###
###############
from enum import Enum

import numpy as np

import cupy as cp

import autograd.numpy as anp
from autograd import grad 

import jax.numpy as jnp
from jax import grad



###################
### Definitions ###
###################

class NUMPY_VERSION(Enum):
    NUMPY = np
    CUPY = cp
    AUTOGRAD = anp
    JAX = jnp

#################
### Functions ###
#################

def numpy_function_retriever(func_name:str, list_params:list=None, 
                             dict_params:dict=None, numpy_version:NUMPY_VERSION=NUMPY_VERSION.NUMPY):
    
    # Check if numpy version is valid
    if not isinstance(numpy_version, NUMPY_VERSION):
        raise ValueError("Invalid numpy_version. Must be a member of NUMPY_VERSION.")
    
    # Retrieve the function from the numpy version
    numpy_lib = numpy_version.value
    try:
        func = getattr(numpy_lib, func_name)  # Dynamically get the function
    except AttributeError:
        raise ValueError(f"The function '{func_name}' does not exist in {numpy_version.name}.")
    
    # Call the function with the provided list and dict parameters
    if list_params and dict_params:
        return func(*list_params, **dict_params)
    elif list_params and not dict_params:
        return func(*list_params)
    elif not list_params and dict_params:
        return func(*dict_params)
    elif not list_params and not dict_params:
        return func()

# def power(number_1, number_2, numpy_version:NUMPY_VERSION):
#     if not isinstance(numpy_version, NUMPY_VERSION):
#         raise ValueError("Invalid numpy_version. Must be a member of NUMPY_VERSION.")
#     return numpy_version.value.power(number_1, number_2)

if __name__ == "__main__":
    nfr = numpy_function_retriever
    print(nfr("array", list_params=[[1,2,3,4,5,6]]))
    print(nfr("array", list_params=[[1,2,3,4,5,6]]).reshape(2, -1))


