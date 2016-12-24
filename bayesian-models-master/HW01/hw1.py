### Daniel Kronovet (dbk2123)
### EECS E6892 HW 01
### October 02, 2015

import os
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import NaiveBayes
from utils import TestHarness, ImageDrawer

random.seed('satyagraha')
path = os.path.dirname(os.path.realpath(__file__))

Xtest = pd.read_csv(path + '/hw1_data_csv/Xtest.csv', header=None)
Xtrain = pd.read_csv(path + '/hw1_data_csv/Xtrain.csv', header=None)
ytest = pd.read_csv(path + '/hw1_data_csv/ytest.csv', header=None, squeeze=True)
ytrain = pd.read_csv(path + '/hw1_data_csv/ytrain.csv', header=None, squeeze=True)
Q = pd.read_csv(path + '/hw1_data_csv/Q.csv', header=None)

# y = 0 -> 4
# y = 1 -> 9
'''
from HW01.hw1 import *
B(A())
nb = NaiveBayes(Xtrain, ytrain)

from HW01.hw1 import *; B(A())
'''

drawer = ImageDrawer(Q)

def A():
    nb = NaiveBayes(Xtrain, ytrain)
    nb.train()
    return nb

def B(nb):
    harness = TestHarness(Xtest, ytest)
    harness.test(nb, print_cm=True)
    return harness

def C():
    drawer = ImageDrawer(Q)
    # Pick three misclassified images.
    # For each image:
    #   Draw x
    #   Print predictive probability

def D(predictions, draw=False):
    confidences = predictions['conf']
    confidences = confidences - confidences.mean()
    confidences = np.abs(confidences / confidences.max())
    confidences.sort()
    most_confusing = confidences.iloc[:3]

    for idx in most_confusing.index:
        print idx, confidences.ix[idx]

    # Interactive, so it can be run separately
    if draw:
        for idx in most_confusing.index:
            drawer.draw_image(Xtest.ix[idx])

    return confidences

if __name__ == '__main__':
    A()

