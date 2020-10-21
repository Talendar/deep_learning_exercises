""" Performance comparison between a MLP model and a RBF network model using UCI's wine data set.

@author Gabriel Nogueira (Talendar)
@author Marcel Otoboni
"""

from mlp.multilayer_perceptron import MultilayerPerceptron
from mlp.cost_functions import MeanSquaredError
from rbf_net.RBFNetwork import RBFNetwork
import pandas as pd
import numpy as np

############### CONFIG ###############
TEST_SET_PC = 0.3                    # percentage of the data to be used to test the model
NORMALIZE_DATA = True                # if true, the data will be normalized before being processed
                                     #
MLP_HIDDEN_LAYERS = [32]             # hidden layers architecture of the MLP
MLP_TRAINING_EPOCHS = 100            # number of training iterations for the MLP
MLP_LEARNING_RATE = 1                # the learning rate for the MLP
                                     #
RBF_NUM_CLUSTERS = 8                 # number of clusters (parameter for the k-means algorithm of the RBF network)
RBF_TRAINING_EPOCHS = 100            # number of training iterations for the RBF network
RBF_LEARNING_RATE = 0.5              # the learning rate for the RBF network
######################################


def load_wine():
    """ Loads the wine data. """
    df = pd.read_csv("./data/wine.data").sample(frac=1).reset_index(drop=True)  # loads and shuffles data
    X, Y = [], []
    for i, row in df.iterrows():
        label, features = int(row["class"]), row.drop("class").values
        X.append(features)

        y = np.zeros(3)
        y[label-1] = 1
        Y.append(y)

    return np.array(X), np.array(Y)


def normalize_data(training_data, test_data):
    """ Normalizes the data. """
    mean, std = np.mean(training_data, axis=0), np.std(training_data, axis=0)
    return (training_data - mean) / std, \
           (test_data - mean) / std


def accuracy(H, Y):
    """ Given a model's predictions and a set of labels, calculates the model's accuracy. """
    hits = 0
    for h, y in zip(H.T, Y):
        h = np.argmax(h, axis=0)
        y = np.argmax(y, axis=0)

        if h == y:
            hits += 1

    return hits / len(Y)


def train_model(model, epochs, lr):
    """ Trains and evaluates the given model. """
    print("\nStarting training session...")
    model.fit(
        data=X_train, labels=Y_train,
        cost_function=MeanSquaredError(),
        epochs=epochs,
        learning_rate=lr,
    )

    train_accuracy = 100 * accuracy(model.predict(X_train), Y_train)
    test_accuracy = 100 * accuracy(model.predict(X_test), Y_test)

    print("\nAccuracy (training set): %.2f%%" % train_accuracy)
    print("Accuracy (test set): %.2f%%\n" % test_accuracy)
    return train_accuracy, test_accuracy


if __name__ == "__main__":
    data, labels = load_wine()

    i = int(len(data) * TEST_SET_PC)
    X_train, Y_train = data[i:], labels[i:]
    X_test, Y_test = data[:i], labels[:i]

    if NORMALIZE_DATA:
        X_train, X_test = normalize_data(X_train, X_test)

    rbf = RBFNetwork(num_output_neurons=3, num_clusters=RBF_NUM_CLUSTERS)
    mlp = MultilayerPerceptron(input_size=13, layers_size=MLP_HIDDEN_LAYERS + [3], layers_activation="sigmoid")

    print("\nTraining set samples: %d (%d%%)" % (len(X_train), 100 * (1 - TEST_SET_PC)))
    print("Test set samples: %d (%d%%)" % (len(X_test), 100 * TEST_SET_PC))

    print("\n################# MULTI-LAYER PERCEPTRON ##################")
    mlp_train_acc, mlp_test_acc = train_model(mlp, MLP_TRAINING_EPOCHS, MLP_LEARNING_RATE)

    print("\n###################### RBF NETWORK ########################")
    rbf_train_acc, rbf_test_acc = train_model(rbf, RBF_TRAINING_EPOCHS, RBF_LEARNING_RATE)

    print(
        "\n################ FINAL RESULTS (accuracy) ###################\n\n" +
        "       |  Train  |  Test\n" +
        "-----------------------------\n" +
        "  MLP  |  %.2f%% |  %03.2f%%\n" % (mlp_train_acc, mlp_test_acc) +
        "-----------------------------\n" +
        "  RBF  |  %.2f%% |  %03.2f%%\n" % (rbf_train_acc, rbf_test_acc)
    )
