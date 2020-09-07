""" Simple implementation of an ADALINE (Adaptive Linear Element).

In this program, the model is used to classify images in two categories: "A" or inverted "A".

@author Gabriel Nogueira (Talendar)
@author Marcel Otoboni
"""

import numpy as np
from PIL import Image
from adaline import Adaline


################### CONFIG ###################
SAMPLES_DIR_A = "./data/a/"                  # directory where the "A" images are
SAMPLES_DIR_A_INV = "./data/a_inverted/"     # directory where the inverted "A" images are
                                             #
SAMPLES_PER_CLASS = 28                       # total number of images per class
TEST_SAMPLES_PER_CLASS = 5                   # number of samples to include in the test set
                                             #
TRAINING_EPOCHS = 10                         # number of iterations of the learning algorithm
LEARNING_RATE = 0.3                          # learning rate of the model
##############################################


def to_feature_vector(img):
    """ Converts the given image to a feature vector containing -1s (white pixels) and 1s (black pixels). """
    return np.array([1 if p == 0 else -1 for p in list(img.getdata())])


if __name__ == "__main__":
    # loading data
    data_a = []
    for i in range(SAMPLES_PER_CLASS):
        data_a.append((
            to_feature_vector(Image.open(SAMPLES_DIR_A + "%d.png" % i)), -1
        ))

    data_a_inv = []
    for i in range(SAMPLES_PER_CLASS):
        data_a_inv.append((
            to_feature_vector(Image.open(SAMPLES_DIR_A_INV + "%d.png" % i)), +1
        ))

    # shuffling (randomizing samples indices)
    np.random.shuffle(data_a)
    np.random.shuffle(data_a_inv)

    # separating data
    test_data = data_a[:TEST_SAMPLES_PER_CLASS] + data_a_inv[:TEST_SAMPLES_PER_CLASS]
    training_data = data_a[TEST_SAMPLES_PER_CLASS:] + data_a_inv[TEST_SAMPLES_PER_CLASS:]

    # training model
    model = Adaline(25)
    model.fit(training_data, epochs=TRAINING_EPOCHS, learning_rate=LEARNING_RATE)

    # testing model
    print("\n" + "#"*20 + "\n")
    print("Testing...", end="")

    hits = 0
    for sample in test_data:
        x, y = sample
        h = model.classify(x)

        if h == y:
            hits += 1

    print(" done!")
    print("Accuracy (test set): %.2f%%" % (100*hits / len(test_data)))
