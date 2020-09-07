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
        self._weights = np.random.uniform(low=-1, high=1, size=input_size)
        self._bias = np.random.uniform(low=-1, high=1)

    def classify(self, x):
        """ Given a feature vector x, returns the class the sample belongs to (according to the model). """
        z = np.inner(self._weights, x) + self._bias
        return 1 if z > 0 else -1

    def fit(self, data, epochs, learning_rate):
        """ Fits the model to the given data using the stochastic gradient descent. """
        for e in range(epochs):
            print("\nEpoch %d/%d..." % (e+1, epochs), end="")
            hits = 0

            for sample in data:
                x, y = sample
                error = y - self.classify(x)

                if error != 0:
                    self._weights += learning_rate * error * x
                    self._bias += learning_rate * error
                else:
                    hits += 1

            print(" done!")
            print("Accuracy: %.2f%%" % (100 * hits / len(data)))

            if hits == len(data):
                print("\nEarly convergence! Finishing...")
                break
