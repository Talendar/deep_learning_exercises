""" Test of a Multi-layer Perceptron, the standard feedforward neural network.

In this program, the network can be trained to:
	- classify the rows of an identity matrix of size n, using log2(n) neurons in one hidden layer;
	- learn the n-dimensional XOR logical function.

@author Gabriel Nogueira (Talendar)
@author Marcel Otoboni
"""

from multilayer_perceptron import MultilayerPerceptron
from cost_functions import MeanSquaredError

import numpy as np
from functools import reduce
from operator import xor


def accuracy(labels, predictions):
	""" Given a model's predictions and a set of labels, calculates the model's accuracy. """
	hits = 0
	for l, p in zip(labels.T, predictions):
		try:
			if len(l) > 1:
				l = np.argmax(l, axis=0)
				p = np.argmax(p, axis=0)
		except TypeError:
			p = np.rint(p)

		if l == p:
			hits += 1

	return hits / len(labels)


def identity_n(nrows, epochs, lr, hidden_size):
	""" Trains the MLP to deal with the auto-encoder problem for a nxn-dimensional ID matrix."""
	print("\n" + 50*'#' + "\n")
	print("Training case: ID Matrix %dx%d." % (nrows, nrows))

	X, Y = np.identity(nrows), np.identity(nrows)
	mlp = MultilayerPerceptron(input_size=nrows, layers_size=[hidden_size, nrows], layers_activation="sigmoid")
	print("Initial accuracy: %.2f%%" % (100 * accuracy(mlp.predict(X), Y)))

	print("\nStarting training session...")
	mlp.fit(
		data=X, labels=Y,
		cost_function=MeanSquaredError(),
		epochs=epochs,
		learning_rate=lr,
	)

	predictions = mlp.predict(X)
	for x, y, h in zip(X, Y, predictions):
		h = np.rint(h)
		print("OUT: {}  |  EXPECTED: {}  ({})".format(np.rint(h), y, "OK" if np.array_equal(h, y) else "WRONG"))

	print("\nTraining session for ID Matrix %dx%d finished." % (nrows, nrows))
	print("MLP architecture: {}".format([nrows] + [hidden_size] + [nrows]))
	print("Final Accuracy : %.2f%%" % (100*accuracy(Y, mlp.predict(X))))


def make_xor_data(num_var):
	""" Creates a data set using the XOR logic function. """
	assert num_var > 1
	X, Y = [], []

	bin_str = "{:0%db}" % num_var
	for n in range(2**num_var):
		bits = [int(c) for c in bin_str.format(n)]
		Y.append(reduce(xor, bits))
		X.append(np.array(bits))

	return np.array(X), np.array(Y)


def xor_train(num_var, epochs, lr, hidden_layers):
	""" Trains an MLP to solve the n-dimensional XOR problem. """
	print("\n" + 50*'#' + "\n")
	print("Training case: XOR with %d variables." % num_var)

	X, Y = make_xor_data(num_var)
	mlp = MultilayerPerceptron(input_size=num_var, layers_size=hidden_layers + [1], layers_activation="sigmoid")
	print("Initial accuracy: %.2f%%" % (100 * accuracy(mlp.predict(X), Y)))

	print("\nStarting training session...")
	mlp.fit(
		data=X, labels=Y,
		cost_function=MeanSquaredError(),
		epochs=epochs,
		learning_rate=lr,
		batch_size=2**num_var,
	)

	predictions = mlp.predict(X)[0]
	for x, y, h in zip(X, Y, predictions):
		print("IN: {}  |  OUT: {:.2f} -> {}  |  EXPECTED: {})".format(x, h, np.rint(h), y))

	print("\nTraining session for %d variables XOR finished." % num_var)
	print("MLP architecture: {}".format([num_var] + hidden_layers + [1]))
	print("Final accuracy : %.2f%%\n" % (100 * accuracy(Y, predictions)))


if __name__ == "__main__":
	opt = -1
	while opt != 2:
		print(
			"\n\n< Multilayer-perceptron Tester >\n" +
			"\t[0] Auto-encoder (ID matrix)\n" +
			"\t[1] XOR\n" +
			"\t[2] Exit\n" +
			"Option: ", end=""
		)
		opt = int(input())

		if opt == 0 or opt == 1:
			print("\nLearning rate: ", end="")
			lr = float(input())

			print("Training epochs: ", end="")
			epochs = int(input())

			# ID(n)
			if opt == 0:
				print("Identity matrix size (n): ", end="")
				n = int(input())

				hidden_size = int(np.ceil(np.log2(n)))
				print("The hidden layer size is log2(n) = %d" % hidden_size)

				identity_n(n, epochs, lr, hidden_size)
			# XOR
			else:
				print("Number of variables for the XOR function: ", end="")
				num_xor = int(input())

				print("Hidden layers: ", end="")
				hidden_layers = [int(i) for i in input().split()]

				xor_train(num_xor, epochs, lr, hidden_layers)

	print("\nLeaving...\n")
