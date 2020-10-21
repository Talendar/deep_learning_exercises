""" Trains and evaluates a neural network (MLP) on one of the two data sets:
        . Wine Data Set (https://archive.ics.uci.edu/ml/datasets/Wine)
        . Geographical Original of Music Data Set (https://archive.ics.uci.edu/ml/datasets/Geographical+Original+of+Music#)

@author Gabriel Nogueira (Talendar)
@author Marcel Otoboni
"""

from mlp.multilayer_perceptron import MultilayerPerceptron
from mlp.cost_functions import MeanSquaredError

import numpy as np
import pandas as pd


############### CONFIG ###############
WINE_DATA = True                     # if true, the wine data set is used; if false, the music data set is used
                                     #
TEST_SET_PC = 0.3                    # percentage of the data to be used to test the model
NORMALIZE_DATA = True                # if true, the input data will be normalized before being processed
NORMALIZE_MUSIC_LABELS = True        # if true, the labels of the music data set will be normalized
                                     #
HIDDEN_LAYERS = [32]                 # hidden layers architecture
TRAINING_EPOCHS = 200                # number of training iterations
LEARNING_RATE = 3                    # the model's learning rate
MOMENTUM_TERM = 0.5                  # value of the momentum term (used to speed up SGD)
######################################


def load_wine():
    """ Loads the Wine Data Set. """
    df = pd.read_csv("./data/wine.data").sample(frac=1).reset_index(drop=True)  # loads and shuffles data
    X, Y = [], []
    for i, row in df.iterrows():
        label, features = int(row["class"]), row.drop("class").values
        X.append(features)

        y = np.zeros(3)
        y[label-1] = 1
        Y.append(y)

    return np.array(X), np.array(Y)


def load_music():
    """ Loads the Geographical Original of Music Data Set. """
    df = pd.read_csv("./data/default_features_1059_tracks.txt", header=None).sample(frac=1).reset_index(drop=True)
    LABELS_COLS = df.columns[-2:]

    X, Y = [], []
    for i, row in df.iterrows():
        label, features = row[LABELS_COLS].values, row.drop(LABELS_COLS).values
        X.append(features)
        Y.append(label)

    return np.array(X), np.array(Y)


def normalize_data(training_data, test_data):
    """ Normalizes the input data. """
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


def abs_error(H, Y):
    """ Evaluates the model trained with the music data set. """
    abs_errors = []
    for h, y in zip(H.T, Y):
        abs_errors.append( np.mean(np.abs(h - y)) )

    return np.mean(abs_errors), np.std(abs_errors)


def evaluate_model():
    """ Evaluates the model on the training and test sets. """
    global mlp, X_train, Y_train, X_test, Y_test, mdiv, WINE_DATA
    if WINE_DATA:
        print("\n[Training set] Accuracy: %.2f%%" % (100 * accuracy(mlp.predict(X_train), Y_train)))
        print("    [Test set] Accuracy: %.2f%%" % (100 * accuracy(mlp.predict(X_test), Y_test)))
    else:
        train_mae, train_std = abs_error(mdiv*mlp.predict(X_train), mdiv*Y_train)
        test_mae, test_std = abs_error(mdiv*mlp.predict(X_test), Y_test)
        print("\n[Training set]  MAE: {:.2f}  |  Abs. error std: {:.2f}".format(train_mae, train_std))
        print("    [Test set]  MAE: {:.2f}  |  Abs. error std: {:.2f}".format(test_mae, test_std))


if __name__ == "__main__":
    # printing settings
    print(
        "\n" +
        "<----------------  SETTINGS  ---------------->\n" +
        " . Data set: %s\n" % ("Wine Data Set" if WINE_DATA else "Geographical Original of Music Data Set") +
        " . Test set pc: %.f%%\n" % (100*TEST_SET_PC) +
        " . Normalize input data: %s\n" % ("yes" if NORMALIZE_DATA else "no") +
        ("" if WINE_DATA else " . Normalize labels (music data set): %s\n" % ("yes" if NORMALIZE_MUSIC_LABELS else "no")) +
        " . Hidden layers: {}\n".format(HIDDEN_LAYERS) +
        " . Training epochs: %d\n" % TRAINING_EPOCHS +
        " . Learning rate: %f\n" % LEARNING_RATE +
        " . Momentum term: %f\n" % MOMENTUM_TERM +
        "---------------------------------------------"
    )

    # preparing data
    data, labels = load_wine() if WINE_DATA else load_music()

    i = int(len(data) * TEST_SET_PC)
    X_train, Y_train = data[i:], labels[i:]
    X_test, Y_test = data[:i], labels[:i]

    print("\nTraining set samples: %d (%d%%)" % (len(X_train), 100 * (1 - TEST_SET_PC)))
    print("Test set samples: %d (%d%%)" % (len(X_test), 100 * TEST_SET_PC))

    # normalizing data
    if NORMALIZE_DATA:
        X_train, X_test = normalize_data(X_train, X_test)

    mdiv = 1
    if not WINE_DATA and NORMALIZE_MUSIC_LABELS:
        mdiv = np.max(Y_train)
        Y_train /= mdiv

    # creating model
    mlp = MultilayerPerceptron(input_size=X_train.shape[1],
                               layers_size=HIDDEN_LAYERS + [Y_train.shape[1]],
                               layers_activation=("sigmoid" if WINE_DATA else "linear"))

    # pre-evaluating
    evaluate_model()

    # training model
    print("\nStarting training session...")
    mlp.fit(
        data=X_train, labels=Y_train,
        cost_function=MeanSquaredError(),
        epochs=TRAINING_EPOCHS,
        learning_rate=LEARNING_RATE,
        momentum_term=MOMENTUM_TERM,
    )

    # evaluating
    print("\nFinal performance on the %s:"
          % ("Wine Data Set" if WINE_DATA else "Geographical Original of Music Data Set"), end="")
    evaluate_model()