### Daniel Kronovet (dbk2123)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRED = 'pred'
CONF = 'conf'

class TestHarness(object):
    def __init__(self, Xtest, ytest):
        self.Xtest = Xtest
        self.ytest = ytest
        self.predictions = None

    def test(self, classifier, print_cm=False):
        self.predictions = classifier.predict(self.Xtest)
        if print_cm:
            self.confusion_matrix(classifier)
        return self.predictions

    def confusion_matrix(self, classifier):
        assert self.predictions is not None, 'Must run predict first!'
        cm = pd.DataFrame(np.zeros((classifier.k, classifier.k)))
        for i in self.ytest.index:
            cm.loc[self.ytest[i], self.predictions.loc[i, PRED]] += 1
        accuracy = sum([cm.loc[i,i] for i in cm.index]) / float(len(self.ytest))
        print 'Accuracy =', accuracy
        print cm

    def misclassified(self):
        assert self.predictions is not None, 'Must run predict first!'
        idx = self.ytest[self.ytest != self.predictions[PRED]].index
        return idx

class ImageDrawer(object):
    def __init__(self, Q):
        self.Q = Q

    def draw_image(self, x):
        image = self.Q.dot(x).reshape(28, 28)
        plt.imshow(image)