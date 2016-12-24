### Daniel Kronovet (dbk2123)
### EECS E6892 HW 02
### October 16, 2015

import os
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import ProbitRegression
from utils import TestHarness, ImageDrawer

random.seed('hiphoppopatamus')
path = os.path.dirname(os.path.realpath(__file__))

Xtest = pd.read_csv(path + '/hw2_data_csv/Xtest.csv', header=None)
Xtrain = pd.read_csv(path + '/hw2_data_csv/Xtrain.csv', header=None)
ytest = pd.read_csv(path + '/hw2_data_csv/ytest.csv', header=None, squeeze=True)
ytrain = pd.read_csv(path + '/hw2_data_csv/ytrain.csv', header=None, squeeze=True)
Q = pd.read_csv(path + '/hw2_data_csv/Q.csv', header=None)

# y = 0 -> 4
# y = 1 -> 9
'''
from HW02.hw2 import *
pr = ProbitRegression(Xtrain, ytrain)
B(A())

from HW02.hw2 import *; pr = A()
from HW02.hw2 import *; B(A())
'''

drawer = ImageDrawer(Q)

def A():
    pr = ProbitRegression(Xtrain, ytrain, lmbda=1, sigma=1.5)
    pr.train()
    return pr

def B(pr):
    harness = TestHarness(Xtest, ytest)
    harness.test(pr, print_cm=True)
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

