### Daniel Kronovet (dbk2123)
### EECS E6892 HW 03
### November 20, 2015

import os
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import VIRegression


random.seed(' ')
path = os.path.dirname(os.path.realpath(__file__))

X1 = pd.read_csv(path + '/data_csv/X_set1.csv', header=None)
X2 = pd.read_csv(path + '/data_csv/X_set2.csv', header=None)
X3 = pd.read_csv(path + '/data_csv/X_set3.csv', header=None)
y1 = pd.read_csv(path + '/data_csv/y_set1.csv', header=None, squeeze=True)
y2 = pd.read_csv(path + '/data_csv/y_set2.csv', header=None, squeeze=True)
y3 = pd.read_csv(path + '/data_csv/y_set3.csv', header=None, squeeze=True)
z1 = pd.read_csv(path + '/data_csv/z_set1.csv', header=None, squeeze=True)
z2 = pd.read_csv(path + '/data_csv/z_set2.csv', header=None, squeeze=True)
z3 = pd.read_csv(path + '/data_csv/z_set3.csv', header=None, squeeze=True)

'''
from HW03.hw3 import *; vir = init(X1, y1)
'''

def transform(z):
    return z.apply(np.sinc) * 10

def init(X, y):
    small = 10 ** -16
    vir = VIRegression(X, y, a=small, b=small, e=1, f=1)
    vir.train()
    return vir

def A(vir):
    plt.plot(vir.loglikelihood)

def B(vir):
    a = vir.a
    alphas = [bk/a for bk in vir.b]
    idx = range(len(alphas))
    markerline, stemlines, baseline = plt.stem(idx, alphas, '-.')
    plt.show()

def C(vir):
    return vir.f / vir.e

def D(vir):
    yhat = vir.X.dot(vir.mu)
    plt.plot(z1, yhat, 'bo-', z1, transform(z1), 'g-')

if __name__ == '__main__':
    A()

