import pandas as pd
import numpy as np
from scipy.special import gammaln, digamma
import matplotlib.pyplot as plt


class VariationalInference(object):
    def __init__(self, X, y, a, b, e, f):
        self.X = X
        self.y = y
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.loglikelihood = []
        self.w = np.zeros(self.d)
        self.sumxixi = np.zeros((self.d, self.d))
        self.sumyixi = np.zeros((self.d, 1))
        for i in range(self.N):
            Xi = X[i].reshape(self.d, 1)
            self.sumxixi += Xi.dot(Xi.T)
            self.sumyixi += self.y[i] * Xi

        self.a0 = float(a)
        self.b0 = float(b)
        self.e0 = float(e)
        self.f0 = float(f)
        self.a = [self.a0 + 0.5] * self.d
        self.b = [self.b0] * self.d
        self.e = self.e0 + self.N / 2
        self.f = self.f0
        self.sigma = np.diag(np.divide([self.a0] * self.d, self.b))
        self.miu = np.zeros((self.d, 1))

    def train(self, iterationNum=500):
        for i in range(iterationNum):
            self.update_qalpha()
            self.update_qlambda()
            self.update_qw()
            self.evaluateObjectFunction()

    def update_qalpha(self):
        for k in range(self.d):
            self.b[k] = 0.5 * (self.sigma + np.dot(self.miu, self.miu.T))[k, k] + self.b0

    def yminusTmu_plus_xsigmax(self):
        self.xsigmax = 0
        self.yminusTmu = 0
        for i in range(self.N):
            Xi = self.X[i].reshape(self.d, 1)
            self.xsigmax += Xi.T.dot(self.sigma).dot(Xi)
            self.yminusTmu += (self.y[i] - self.miu.T.dot(self.X[i])) ** 2

    def update_qlambda(self):
        self.f = self.f0
        self.yminusTmu_plus_xsigmax()
        self.f += 0.5 * (self.yminusTmu + self.xsigmax)

    def update_qw(self):
        self.sigma = np.linalg.inv(np.diag(np.divide(self.a, self.b)) + (self.e / self.f) * self.sumxixi)
        self.miu = self.sigma.dot((self.e / self.f) * self.sumyixi)

    def _log_determinant(self, M):
        L = np.linalg.cholesky(M)
        L_inv = np.linalg.inv(L)
        return 2 * sum([np.log(el) for el in np.diag(L_inv)])

    def evaluateObjectFunction(self):
        e0, e, f0, f = self.e0, self.e, self.f0, self.f
        a0, a, b0, b = self.a0, self.a[0], self.b0, self.b
        N, d, sigma, miu = self.N, self.d, self.sigma, self.miu
        self.yminusTmu_plus_xsigmax()
        term1 = e0 * np.log(f0) - gammaln(e0) - (e0 - e) * (digamma(e) - np.log(f)) - (f0 - f) * (e / f) - e * np.log(
            f) + gammaln(e)
        term2 = sum([(digamma(a) - np.log(bk)) for bk in b]) - 0.5 * sum(
            [(sigma + miu.dot(miu.T))[k, k] * a / b[k] for k in range(len(b))]) + 0.5 * self._log_determinant(
            sigma) + d / 2.0
        term3 = d * (a0 * np.log(b0) - gammaln(a0)) - (a * sum([np.log(bk) for bk in b]) + d * gammaln(a)) + (
                                                                                                             a0 - a) * sum(
            [(digamma(a) - np.log(bk)) for bk in b]) - (b0 * sum([a / bk for bk in b])) + d * a
        term4 = N / 2.0 * (digamma(e) - np.log(f)) - N / 2.0 * np.log(2 * np.pi) - f / (2 * e) * (
        self.yminusTmu + self.xsigmax)
        l = int(term1 + term2 + term3 + term4)
        self.loglikelihood.append(l)

def plotObjectFunc(model, dataset_index):
    myplot = plt.figure()
    plt.plot(model.loglikelihood)
    plt.title('Variational Objective Function across iterations for data set '+str(dataset_index))
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel('Variational Objective Function')
    myplot.savefig('HMWK3_Q2_A'+str(dataset_index)+'.png')


def plotexpectationoveralpha(model, dataset_index):
    myplot = plt.figure()
    alpha = np.divide(model.a, model.b)
    plt.stem( 1/alpha )
    plt.title(r'$1/ E[\alpha_k]$ as a function of k for data set ' +str(dataset_index))
    plt.grid()
    plt.xlabel('k')
    plt.ylabel(r'$1/ E[\alpha_k]$')
    myplot.savefig('HMWK3_Q2_B'+str(dataset_index)+'.png')

def plotcomparisonlines(x,y,z,model,dataset_index):
    myplot = plt.figure()
    yhat = x.dot(model.miu)
    z_n = np.linspace(-6,6,len(y))
    fz_n = 10 * np.sinc(z_n)
    plt.plot(z,yhat,color='red',label='Predicted')
    plt.scatter(z,y,color='green',label='Actual')
    plt.plot(z_n,fz_n,label='True')
    plt.legend()
    plt.title('Actual, Predicted, and True function for data set '+str(dataset_index))
    plt.xlabel('$z_{i}$ $&$ $f(z_i)$')
    plt.ylabel('$y_{i}$ $&$ $\hat{y_{i}}$')
    plt.grid()
    myplot.savefig('HMWK3_Q2_D'+str(dataset_index)+'.png')



X1 = pd.read_csv('data_csv/X_set1.csv', header=None).values
X2 = pd.read_csv('data_csv/X_set2.csv', header=None).values
X3 = pd.read_csv('data_csv/X_set3.csv', header=None).values
y1 = pd.read_csv('data_csv/y_set1.csv', header=None, squeeze=True).values
y2 = pd.read_csv('data_csv/y_set2.csv', header=None, squeeze=True).values
y3 = pd.read_csv('data_csv/y_set3.csv', header=None, squeeze=True).values
z1 = pd.read_csv('data_csv/z_set1.csv', header=None, squeeze=True).values
z2 = pd.read_csv('data_csv/z_set2.csv', header=None, squeeze=True).values
z3 = pd.read_csv('data_csv/z_set3.csv', header=None, squeeze=True).values

for i in range(3):
    if i == 0:
        X, y, z = X1, y1, z1
    elif i == 1:
        X, y, z = X2, y2, z2
    else:
        X, y, z = X3, y3, z3

    model = VariationalInference(a=10**-16, b=10**-16, e=1, f=1, X=X, y=y)
    model.train()
    plotObjectFunc(model, i+1)
    plotexpectationoveralpha(model, i+1)
    print(model.f/model.e)
    plotcomparisonlines(X, y, z, model, i+1)


