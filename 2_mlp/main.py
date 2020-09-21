""" Implementation and test of a MLP network

In this program the network was trained to classify the identity matrix of size n, and a case for XOR local operator of size 2

In this program, the model is used to classify images in two categories: "A" or inverted "A".
@author Gabriel Nogueira (Talendar)
@author Marcel Otoboni
"""

from multilayer_perceptron import MultilayerPerceptron
from cost_functions import MeanSquaredError

import numpy as np

############### CONFIG ###############
TRAINING_EPOCHS = 100                # number of training iterations
LEARNING_RATE = 10                    # the model's learning rate
######################################

def accuracy(H, Y):
	""" Given a model's predictions and a set of labels, calculates the model's accuracy. """
	hits = 0
	for h, y in zip(H.T, Y):
		h = np.argmax(h, axis=0)
		y = np.argmax(y, axis=0)

		if h == y:
			hits += 1

	return hits / len(Y)

def identity_n(nrows, epochs=TRAINING_EPOCHS):
	#nrows is the size of identity matrix
	if(float(nrows).is_integer() == False):
		return None

	X, Y = np.identity(nrows), np.identity(nrows)
	
	print(50*'#')
	print("Training case of matrix identity = %d" % (nrows))

	mlp = MultilayerPerceptron(input_size=nrows, layers_size=[int(np.ceil(np.log2(nrows)))] + [nrows], layers_activation="sigmoid")
	print("\nInitial accuracy (training set): %.2f%%" % (100 * accuracy(mlp.predict(X), Y)))

	print("\nStarting training session...")
	mlp.fit(
	data=X, labels=Y,
	cost_function=MeanSquaredError(),
	epochs=epochs,
	learning_rate=LEARNING_RATE,
	batch_size=8,
	gradient_checking=False
	)

	print("Accuracy : %.2f%%\n" % (100*accuracy(mlp.predict(X), Y)))
	return mlp

def xor_train(epochs=TRAINING_EPOCHS):
	print(50*'#')
	print("XOR case")
	xor_X, xor_Y = np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,0])

	xor_mlp = MultilayerPerceptron(input_size=2, layers_size=[4] + [1], layers_activation="sigmoid")
	print("\nInitial accuracy (training set): %.2f%%" % (100 * accuracy(xor_mlp.predict(xor_X), xor_Y)))

	print("\nStarting training session...")
	xor_mlp.fit(
		data=xor_X, labels=xor_Y,
		cost_function=MeanSquaredError(),
		epochs=epochs,
		learning_rate=LEARNING_RATE,
		batch_size=4,
		gradient_checking=False
	)

	print("Accuracy : %.2f%%\n" % (100*accuracy(xor_mlp.predict(xor_X), xor_Y)))
	return xor_mlp

if __name__ == "__main__":
	
	identity_n(8, epochs=1000)
	
	identity_n(15, epochs=1000)
	
	
	xor_train(epochs=1000)
	#test_print()