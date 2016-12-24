### Daniel Kronovet (dbk2123)
### EECS E6892 HW 04
### December 11, 2015

import os
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import GMM_EM, GMM_VI, GMM_Gibbs


random.seed('Goethe')
path = os.path.dirname(os.path.realpath(__file__))

X = pd.read_csv(path + '/data.txt', header=None)

'''
from HW04.hw4 import *; gmm = A(4)
from HW04.hw4 import *; gmm = B(4)
from HW04.hw4 import *; gmm = C()
'''

def A(k=4):
    gmm = GMM_EM(X, k=k)
    gmm.train()
    return gmm

def B(k=4):
    gmm = GMM_VI(X, k=k, alpha=1, c=10)
    gmm.train()
    return gmm

def C():
    gmm = GMM_Gibbs(X, alpha=1, c=.1)
    gmm.train()
    return gmm

if __name__ == '__main__':
    pass
