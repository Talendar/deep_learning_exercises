# some_file.py
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../2_mlp')

from multilayer_perceptron import MultilayerPerceptron
from cost_functions import MeanSquaredError

import numpy as np
import pandas as pd
#import pygame

############### CONFIG ###############
TEST_SET_PC = 0.2                    # percentage of the data to be used to test the model
HIDDEN_LAYERS = [64]             # hidden layers architecture
TRAINING_EPOCHS = 100                # number of training iterations
LEARNING_RATE = 0.03                   # the model's learning rate
######################################

def load_mnist(path):
    """ Loads and shuffles the MNIST data. """
    df = pd.read_csv(path).sample(frac=1).reset_index(drop=True)  # loads and shuffles data
    X, Y = [], []

    for i, row in df.iterrows():
        label, pixels = row["label"], row.drop("label").values / 255
        X.append(pixels)

        y = np.zeros(10)
        y[label] = 1
        Y.append(y)

    return np.array(X), np.array(Y)

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', 
    sep=',', 
    header=None
)
labels = df.values[:,:1]
data = np.array([[round(j, 4).astype(float) for j in i] for i in df.values[:,1:]])

print(data)


def accuracy(H, Y):
    """ Given a model's predictions and a set of labels, calculates the model's accuracy. """
    hits = 0
    for h, y in zip(H.T, Y):
        h = np.argmax(h, axis=0)
        y = np.argmax(y, axis=0)

        if h == y:
            hits += 1

    return hits / len(Y)




print(len(data))
i = int(len(data) * TEST_SET_PC)
X_train, Y_train = data[i:].astype(float), labels[i:].astype(int)
X_test, Y_test = data[:i].astype(float), labels[:i].astype(int)

print("\nTraining set samples: %d (%d%%)" % (len(X_train), 100*(1 - TEST_SET_PC)))
print("Test set samples: %d (%d%%)" % (len(X_test), 100*TEST_SET_PC))

mlp = MultilayerPerceptron(input_size=13, layers_size=HIDDEN_LAYERS + [3], layers_activation="sigmoid")
print("\nInitial accuracy (training set): %.2f%%" % (100 * accuracy(mlp.predict(X_train), Y_train)))
print("Initial accuracy (test set): %.2f%%" % (100 * accuracy(mlp.predict(X_test), Y_test)))






print("\nStarting training session...")
mlp.fit(
  data=X_train, labels=Y_train,
  cost_function=MeanSquaredError(),
  epochs=TRAINING_EPOCHS,
  learning_rate=LEARNING_RATE,
  batch_size=32,
  gradient_checking=False
)

print("\nAccuracy (training set): %.2f%%" % (100*accuracy(mlp.predict(X_train), Y_train)))
print("Accuracy (test set): %.2f%%\n" % (100*accuracy(mlp.predict(X_test), Y_test)))