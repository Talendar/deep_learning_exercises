""" Implementation of a Radial Basis Function (RBF) Network.

@author Gabriel Nogueira (Talendar)
@author Marcel Otoboni
"""

from mlp.multilayer_perceptron import MultilayerPerceptron
import numpy as np
from sklearn.cluster import KMeans


class RBFNetwork:
    """ Implementation of a Radial Basis Function (RBF) Network. """

    def __init__(self, num_output_neurons, num_clusters=8):
        """ Instantiates a new RBF network.

        :param num_output_neurons: number of neurons in the output layer.
        :param num_clusters: number of clusters to be considered by the k-means algorithm.
        """
        self._num_clusters = num_clusters
        self._kmeans = None
        self._mlp = MultilayerPerceptron(num_clusters, layers_size=[num_output_neurons], layers_activation="linear")

    def _gauss_rbf(self, data, breadth_param=1):
        """ Transforms the data using the Gaussian radial basis function. """
        transformed = []
        for x in data:
            trans_x = np.zeros(self._num_clusters)
            for i, u in enumerate(self._kmeans.cluster_centers_):       # iterate through centroids of the clusters
                v = np.linalg.norm(x - u)                               # distance between x and the centroid
                trans_x[i] = np.exp(-(v**2) / (2 * breadth_param**2))   # gaussian function
            transformed.append(trans_x)
        return np.array(transformed)

    def predict(self, x, just_one=False):
        """ Predicts the class of the given example. """
        if just_one:
            x = np.array([x])
        return self._mlp.predict(self._gauss_rbf(x))

    def fit(self, data, labels, cost_function, epochs, learning_rate):
        """ Fits the model to the given data.

        :param data: numpy 2D-array in which each ROW represents an input sample vector.
        :param labels: numpy 2D-array in which each ROW represents a vector with the samples' labels.
        :param cost_function: cost function to be minimized.
        :param epochs: number of training epochs (iterations).
        :param learning_rate: learning rate of the model.
        """
        self._kmeans = KMeans(n_clusters=self._num_clusters).fit(data)
        data = self._gauss_rbf(data)
        self._mlp.fit(data, labels, cost_function, epochs, learning_rate)
