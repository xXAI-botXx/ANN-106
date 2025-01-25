"""
Loss Functions should be defined as a string where every element is splitted with one space.

"""

import numpy as np

import autograd.numpy as anp
from autograd import grad 

import jax.numpy as jnp
from jax import grad

# import sympy as sp
# import statistics
from enum import Enum

###################
### Definitions ###
###################

class REALIZATION_WAYS(Enum):
    JAX = 1
    AUTOGRAD = 2

# class FUNCTION_SYMBOLS(Enum):
#     # Structures
#     START = "("
#     END = ")"
    
#     # Functions -> need START ( and END )
#     SUM = "SUM"
#     MEAN = "MEAN"
#     MEDIAN = "MEDIAN"
#     ROOT = "ROOT"

#     # Operations
#     ADD = "+"
#     SUB = "-"
#     MULTI = "*"
#     DEVISION = "/"
#     MODULO = "%"
#     EXP = "**"

#     # Constants
#     PI = "PI"
#     E = "E"

#     # Variables
#     Y_PREDICT = "Y_PREDICT"
#     Y_REAL = "Y_REAL"
#     # VAR1 = "VAR1"
#     # VAR2 = "VAR2"
#     # VAR3 = "VAR3"
#     # VAR4 = "VAR4"

# # set another name, for easy access
# F = FUNCTION_SYMBOLS
# NEED_START_END = [F.SUM.value, F.MEAN.value, F.MEDIAN.value, F.ROOT.value]

# SYMPY_SYMBOL_MAPPING = {

#     # Functions
#     FUNCTION_SYMBOLS.SUM.value: sp.Add,
#     FUNCTION_SYMBOLS.MEAN.value: lambda *args: sp.Add(*args) / len(args),  # Mean as average
#     FUNCTION_SYMBOLS.MEDIAN.value: lambda *args: statistics.median(args),  # Median using statistics
#     FUNCTION_SYMBOLS.ROOT.value: sp.sqrt,

#     # Constants
#     FUNCTION_SYMBOLS.PI.value: sp.pi,
#     FUNCTION_SYMBOLS.E.value: sp.E,

#     # Variables
#     FUNCTION_SYMBOLS.Y_PREDICT.value: sp.symbols('Y_PREDICT'),
#     FUNCTION_SYMBOLS.Y_REAL.value: sp.symbols('Y_REAL'),
# }


#############
### Tools ###
#############

def build_function(*function_elements):
    """
    Builds a string function from multiple single function elements.

    Connects and extracts the single values.

    Very helpful fr easily build a function.

    :param function_elements: Multiple elements each as single parameter and 

    :example:
    
    >>> function_builder(F.SUM, F.START, F.Y_PREDICT, F.ADD, 12, F.END)
    >>> 'SUM ( Y_PREDICT + 12 )'
    """
    extracted_elements = []
    for cur_elem in function_elements:
        if isinstance(cur_elem, FUNCTION_SYMBOLS):
            extracted_elements += [cur_elem.value]
        else:
            # should be number -> int or float
            if isinstance(cur_elem, (int, float)):
                extracted_elements += [cur_elem]
    return " ".join(extracted_elements)

def realize_loss_function(func:str, way:REALIZATION_WAYS=REALIZATION_WAYS.SYMPY):
    if way == REALIZATION_WAYS.SYMPY:
        return ann106_function_to_sympy_function(func)
    elif way == REALIZATION_WAYS.JAX:
        pass
    elif way == REALIZATION_WAYS.AUTOGRAD:
        pass
    else:
        raise ValueError("The way to realize the loss function does not exist!")

def get_start_end_area(content:str):
    """
    Finds the closing bracket in a string. The string have to start with the opening bracket.
    """
    content = content[1:]
    inner_content = ""
    end_counter = 0
    for i in content:
        if i == ")" and end_counter <= 0:
            return inner_content, content[len(inner_content)+1:]
        elif i == ")" and end_counter > 0:
            inner_content += i
            end_counter -= 1
        elif i == "(":
            inner_content += i
            end_counter += 1
        else:
            inner_content += i

    raise ValueError("The amounts of '(' and ')' does not fit!")

# def add_(expr, elem_to_add):
#     if expr:
#         return expr + elem_to_add
#     else:
#         return elem_to_add

# # Works????
# def ann106_function_to_sympy_function(ann106_func:str, sympy_expr=None):

#     if len(ann106_func) <= 0:
#         return sympy_expr
#     else:
#         # get single function parts
#         function_parts = ann106_func.split(" ")
#         cur_part = function_parts[0]
#         rest_parts = function_parts[1:] if len(function_parts) > 0 else None

#         if cur_part in NEED_START_END or cur_part == FUNCTION_SYMBOLS.START.value:
#             if cur_part == FUNCTION_SYMBOLS.START.value:
#                 inner_part, outer_part = get_start_end_area(function_parts)

#                 inner_part_expr = sp.sympify((ann106_function_to_sympy_function(inner_part, sympy_expr)))
#                 outer_part_expr = ann106_function_to_sympy_function(outer_part)
#             else:
#                 inner_part, outer_part = get_start_end_area(rest_parts)
#                 cur_part = SYMPY_SYMBOL_MAPPING[cur_part]

#                 inner_part_expr = cur_part(ann106_function_to_sympy_function(inner_part, sympy_expr))
#                 outer_part_expr = ann106_function_to_sympy_function(outer_part)
#             return inner_part_expr if outer_part_expr is None else inner_part_expr + outer_part_expr
#         else:
#             if cur_part in SYMPY_SYMBOL_MAPPING.keys():
#                 cur_part = SYMPY_SYMBOL_MAPPING[cur_part]
#                 sympy_expr = add_(sympy_expr, cur_part)
#             else:
#                 if cur_part in [FUNCTION_SYMBOLS.START.value, FUNCTION_SYMBOLS.END.value]:
#                     sympy_expr = add_(sympy_expr, cur_part)
#                 else:
#                     sympy_expr = add_(sympy_expr, sp.sympify(cur_part))

#             if rest_parts:
#                 return sympy_expr + ann106_function_to_sympy_function(" ".join(rest_parts))
#             else:
#                 return sympy_expr

def get_total_loss(loss_dict):
    """
    Changes a dict of losses to a total loss.
    """
    return sum([value for value in loss_dict.values()])


######################
### Loss Functions ###
######################

sum_error = build_function(F.SUM, F.START, F.Y_REAL, F.SUB, F.Y_PREDICT, F.END)
# def sum_error(y, y_):
#     return np.sum(y - y_)


def sum_absolute_error(y, y_):
    return np.sum( np.abs(y - y_) )


def mean_absolute_error(y, y_):
    return np.mean( np.abs(y - y_) )


def mean_squared_error(y, y_):
    return np.mean( (y - y_)**2 )


def mean_root_squared_error(y, y_):
    return np.sqrt( mean_squared_error(y, y_) )


def huber_loss(y, y_):
    delta = 1.0
    residuals = y - y_
    huber_loss = np.where(
        np.abs(residuals) <= delta,
        0.5 * residuals**2,
        delta * np.abs(residuals) - 0.5 * delta**2
    )
    return np.mean(huber_loss)

def cost_function(y, y_):
    return 0.5 * np.power(y_ - y, 2)


if __name__ == "__main__":
    print(ann106_function_to_sympy_function("Y_PREDICT ** 3 + 2 * Y_PREDICT ** 2 + 5 * Y_PREDICT + 7"))

    # print(get_start_end_area("(hey, das ist ein test)"))
    # print(get_start_end_area("(hey, das ist ein test) oder?"))
    # print(get_start_end_area("(hey, das ist ein test (der kompleziert ist!)) oder?"))

    # print(ann106_function_to_sympy_function("SUM ( SQRT ( 1 * 2 / 12 + Y_PREDICT ) )"))
    # print(ann106_function_to_sympy_function("SQRT ( SUM ( 1 * 2 / 12 + Y_PREDICT ) )"))

    # print(sum_error)
    # print(realize_loss_function(sum_error))

    # sum_error = build_function(FUNCTION_SYMBOLS.SUM, FUNCTION_SYMBOLS.START, F.Y_REAL, F.SUB, F.Y_PREDICT, FUNCTION_SYMBOLS.END)
    # print("Generierte Funktionszeichenkette:", sum_error)

    # # Umsetzen der Funktion in ein sympy-Objekt
    # sympy_expr = realize_loss_function(sum_error, way=None)
    # print("Sympy Funktion:", sympy_expr)

    # # Beispiel: Eine Funktion mit MEAN und ROOT erstellen
    # mean_root_error = build_function(FUNCTION_SYMBOLS.MEAN, FUNCTION_SYMBOLS.START, F.Y_REAL, F.ADD, F.Y_PREDICT, FUNCTION_SYMBOLS.END)
    # print("Generierte Funktionszeichenkette mit MEAN:", mean_root_error)

    # # Umsetzen der MEAN- und ROOT-Funktion
    # sympy_expr_mean_root = realize_loss_function(mean_root_error, way=None)
    # print("Sympy Funktion mit MEAN und ROOT:", sympy_expr_mean_root)

    # # Beispiel: MEDIAN als Liste von Werten
    # median_example = build_function(FUNCTION_SYMBOLS.MEDIAN, FUNCTION_SYMBOLS.START, 1, 5, 3, 8, 2, FUNCTION_SYMBOLS.END)
    # print("Generierte Funktionszeichenkette mit MEDIAN:", median_example)

    # # Umsetzen der MEDIAN-Funktion
    # sympy_expr_median = realize_loss_function(median_example, way=None)
    # print("Sympy Funktion mit MEDIAN:", sympy_expr_median)



