

def heaviside(sum_value):
    """
    Simple linear step function

    Input: sum of weighted neurons multiplied by the input
    Output: 0 or 1
    """
    if sum_value >= 0:
        return 1
    else:
        return 0



