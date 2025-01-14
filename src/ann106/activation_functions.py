

def pass_through(sum_value):
    return sum_value

def step_function(sum_value, threshold=0.0, greater_equal_value=1.0, smaller_value=0.0):
    """
    Simple linear step function

    Input: sum of weighted neurons multiplied by the input
    Output: 0 or 1
    """
    if sum_value >= threshold:
        return greater_equal_value
    else:
        return smaller_value



