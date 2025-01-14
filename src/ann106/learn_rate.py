
import numpy as np

class LearnrateScheduler():
    """
    Defines the learnrate (learn adjustment) and the strategy of the learnrate adjustment.

    :param epochs: Defines the epochs of the training.
    :type epochs: int
    :param max_steps: Defines the max steps of the training.
    :type max_steps: int
    :param start_learnrate: Sets the initial learnrate
    :type start_learnrate: float
    """
    def __init__(self, start_learnrate=1.0):
        self.start_learn_rate = start_learnrate
        self.learn_rate = start_learnrate

    def step(self):
        """
        Adjusts the learnrate.
        """
        pass




