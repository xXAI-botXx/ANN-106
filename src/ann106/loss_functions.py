

import numpy as np

def get_total_loss(loss_dict):
    """
    Changes a dict of losses to a total loss.
    """
    return sum([value for value in loss_dict.values()])

def sum_error(y, y_):
    return np.sum(y - y_)


def sum_absolute_error(y, y_):
    return np.sum( np.abs(y - y_) )


def mean_absolute_error(y, y_):
    return np.mean(np.abs(y - y_))


def mean_squared_error(y, y_):
    return np.mean( (y - y_pred)**2 )


def huber_loss(y, y_):
    delta = 1.0
    residuals = y - y_pred
    huber_loss = np.where(
        np.abs(residuals) <= delta,
        0.5 * residuals**2,
        delta * np.abs(residuals) - 0.5 * delta**2
    )
    return np.mean(huber_loss)


