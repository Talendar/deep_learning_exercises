""" Simple implementation of an ADALINE (Adaptive Linear Element).

@author Gabriel Nogueira (Talendar)
@author Marcel Otoboni
"""

import numpy as np


class Adaline:
    """ Implementation of the Adaptive Linear Element (ADALINE), an early single-layer artificial neural network.

    Attributes:
        _weights: vector containing the model's weights.
        _bias: real number representing the bias of the model.
    """

    def __init__(self, input_size):
        """ Initializes a new model, adjusted to receive inputs with the given size. """
        self._weights = None  # todo
        self._bias = None  # todo

    def classify(self, x):
        """ Given a feature vector x, returns the class the sample belongs to (according to the model). """
        raise NotImplementedError()  # todo

    def fit(self, data, epochs, learning_rate):
        """ Fits the model to the given data using the stochastic gradient descent. """
        raise NotImplementedError()  # todo
