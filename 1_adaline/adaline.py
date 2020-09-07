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
        self._weights = np.array([0 for i in range(input_size)])
        self._bias = 0.1

    def classify(self, x):
        """ Given a feature vector x, returns the class the sample belongs to (according to the model). """
        return 1 if np.dot(x, self._weights) > 0 else -1

    def fit(self, data, epochs, learning_rate):
        """ Fits the model to the given data using the stochastic gradient descent. """
        for i in range(epochs):
            hits = 0

            for letter in data:
                error = letter[1] - self.classify(letter[0])

                if(error != 0):
                    self._weights = self._weights + letter[0] * error * learning_rate

                else:
                    hits += 1
            
            if(hits == len(data)):
                break
                