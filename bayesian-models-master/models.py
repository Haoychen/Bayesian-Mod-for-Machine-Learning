### Daniel Kronovet (dbk2123)

import math
import random
random.seed('******')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln, digamma
from scipy.stats import norm, multivariate_normal, wishart

PRED = 'pred'
CONF = 'conf'

class BaseModel(object):
    def train(self):
        raise NotImplemented('Must implement!')

    def predict(self, Xhat):
        raise NotImplemented('Must implement!')

    def _init_predictions(self, Xhat):
        predictions = pd.DataFrame(
            np.zeros((len(Xhat),2)),
            index=Xhat.index, columns=[PRED, CONF])
        predictions[PRED] = float('NaN')
        return predictions

    # General math utilities
    def _log_determinant(self, M):
        '''M is any matrix'''
        L = np.linalg.cholesky(M)
        L_inv = np.linalg.inv(L)
        return 2 * sum([np.log(el) for el in np.diag(L_inv)])

# HW 1: Naive Bayes
class NaiveBayes(BaseModel):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert len(X) == len(y)
        self.N = len(X)

        self.categories = set(y)
        self.k = len(self.categories)
        self.d = len(X.iloc[0])
        self.Xparameters = self._init_Xparameters()
        self.yparameters = self._init_yparameters()

    def _init_Xparameters(self):
        param_names = ['m', 'a', 'b', 'c']
        param_idx = pd.DataFrame(np.ones((self.k,self.d))).stack().index
        parameters = pd.DataFrame(
            np.zeros((self.d*self.k, len(param_names))),
            columns=param_names, index=param_idx)
        parameters['m'] = 0
        return parameters

    def _init_yparameters(self):
        param_names = ['pi', 'e', 'f']
        parameters = pd.DataFrame(
            np.ones((self.k, len(param_names))), columns=param_names)
        parameters['pi'] = 0.5
        return parameters

    def train(self):
        for category in self.categories:
            self._train_category(category)

    def _train_category(self, c):
        yparams = self.yparameters.ix[c]
        Xparams = self.Xparameters.ix[c]

        ycategory = self.y[self.y == c]
        Xcategory = self.X.ix[ycategory.index]

        # Train y parameters
        e = yparams['e']
        f = yparams['f']
        ef = e if c == 1 else f
        pi = (ef + len(ycategory))/float(self.N + e + f)
        yparams['pi'] = pi

        # Train d X parameters
        astar, bstar, cstar, mstar = self._update_Xparameters(Xcategory, Xparams)
        self.Xparameters.loc[c,'a'] = list(astar)
        self.Xparameters.loc[c,'b'] = list(bstar)
        self.Xparameters.loc[c,'c'] = list(cstar)
        self.Xparameters.loc[c,'m'] = list(mstar)

    # operating on a matrix
    def _update_Xparameters(self, data, params):
        n = len(data)
        xbar = data.mean()
        s = data.var()
        a = params['a']
        b = params['b']
        c = params['c']
        m = params['m']

        astar = a + n
        bstar = b + n/2.
        cstar = c + ((n*a * (xbar - m)**2)/(a+n) + n*s)/2.
        mstar = (a*m + n*xbar)/(a+n)

        return astar, bstar, cstar, mstar

    def predict(self, Xhat):
        predictions = self._init_predictions(Xhat)

        for c in self.categories:
            confidence = self._predict_category(Xhat, c)
            idx = predictions[predictions[CONF] < confidence].index
            predictions.loc[idx, PRED] = c
            predictions.loc[idx, CONF] = confidence.ix[idx]

        return predictions

    def _predict_category(self, Xhat, c):
        py = self.yparameters.ix[c]['pi']
        xparams = self.Xparameters.ix[c]
        prediction = self._tdistribution(xparams, Xhat)
        return prediction.T.prod() * py

    def _tdistribution(self, params, Xhat):
        a, b, c, m = params['a'], params['b'], params['c'], params['m'] # Series
        bprime = b + 0.5
        cprime = c + (1/2. * (a/(a+1)) * (Xhat-m)**2)

        term1 = math.e**(gammaln(bprime) - gammaln(b)) # Use ln to avoid overflow
        term2 = np.sqrt(a / (2*math.pi * (a+1)))
        term3 = math.e**(b*np.log(c) - (bprime)*np.log(cprime))

        return term1*term2*term3


# HW 2: Expectation-Maximization
class ProbitRegression(BaseModel):
    def __init__(self, X, y, lmbda=1, sigma=1.5):
        self.X = X
        self.y = y
        self.sigma = float(sigma)
        self.lmbda = float(lmbda)
        assert len(X) == len(y)
        self.N = len(X)
        self.d = len(X.T)
        self.k = 2 # 0/1 decision
        self.ephi = pd.Series(np.zeros(self.N))
        self.w = pd.Series(np.zeros(self.d))
        self.loglikelihood = []

    def train(self):
        self.iterate(n=100)

    def predict(self, Xhat):
        predictions = self._init_predictions(Xhat)
        predictions[PRED] = 0
        predictions[CONF] = norm.cdf(Xhat.dot(self.w)/self.sigma)
        cutoff = random.random()
        idx = predictions.loc[predictions[CONF] > 0.5].index
        predictions.loc[idx, PRED] = 1
        return predictions

    def iterate(self, n=1):
        for _ in xrange(n):
            self._iterate()

    def _iterate(self):
        self.Estep()
        self.Mstep()
        self.calcloglikelihood()

    def Estep(self):
        xdotw = self.X.dot(self.w) # Using value of w_t
        sig = self.sigma
        self.ephi[self.y == 1] = xdotw + sig * (norm.pdf(-xdotw/sig)) / (1 - norm.cdf(-xdotw/sig))
        self.ephi[self.y == 0] = xdotw + sig * -(norm.pdf(-xdotw/sig)) / norm.cdf(-xdotw/sig)

    def Mstep(self):
        term1 = self.X.mul(self.ephi, axis=0).sum() / self.sigma**2 # Vector
        term2 = self.lmbda + (self.X.T.dot(self.X) / self.sigma**2) # Matrix
        self.w = term1.dot(np.linalg.inv(term2)) # Value of w_t+1

    def calcloglikelihood(self):
        xdotw = self.X.dot(self.w) # Using value of w_t+1
        term1 = self.d/2. * np.log(self.lmbda/(2*np.pi))
        term2 = self.lmbda/2. * self.w.dot(self.w)
        term3 = self.y.mul(np.log(norm.cdf(xdotw / self.sigma)), axis=0).sum()
        term4 = (1 - self.y).mul(np.log(1 - norm.cdf(xdotw / self.sigma)), axis=0).sum()
        ll = term1 - term2 + term3 + term4
        self.loglikelihood.append(ll)


# HW 3: Variational Inference
class VIRegression(BaseModel):
    def __init__(self, X, y, a=1, b=1, e=1, f=1):
        self.X = X
        self.y = y
        assert len(X) == len(y)
        self.loglikelihood = []
        self.N = len(X)
        self.d = len(X.T)
        self.w = pd.Series(np.zeros(self.d))

        # Distribution Parameters
        self.a0 = float(a)
        self.b0 = float(b)
        self.e0 = float(e)
        self.f0 = float(f)
        self.a = self.a0 + 1/2.
        self.b = [self.b0 for _ in xrange(self.d)]
        self.e = self.e0 + self.N/2.
        self.f = self.f0
        self.mu = pd.DataFrame(np.zeros(self.d))
        self.Sigma = pd.DataFrame(np.diag([self.b0/self.a0 for _ in xrange(self.d)]))

    def train(self):
        self.iterate(n=500)

    def predict(self, Xhat):
        predictions = self._init_predictions(Xhat)
        predictions[PRED] = 0
        predictions[CONF] = norm.cdf(Xhat.dot(self.w)/self.Sigma)
        cutoff = random.random()
        idx = predictions.loc[predictions[CONF] > 0.5].index
        predictions.loc[idx, PRED] = 1
        return predictions

    def iterate(self, n=1):
        for i in xrange(n):
            if not i % 10:
                print 'Iteration %i, f = %.2f' % (i, self.f)
            self._iterate()

    def _iterate(self):
        self.update_qalpha()
        self.update_qlambda()
        self.update_qw()
        self.calcloglikelihood()
        # Putting qw last (instead of first) helped tremendously.

    def update_qalpha(self):
        b0, Sigma, mu = self.b0, self.Sigma, self.mu
        ExxT = self._ExxT()
        for i in xrange(self.d):
            self.b[i] = b0 + (ExxT.iloc[i,i] / 2.) # [0] to get value out of DF.

    def update_qlambda(self):
        f0, Sigma, mu, X, y = self.f0, self.Sigma, self.mu, self.X, self.y
        self.f = f0 + (self._sum_y_minus_xTw_squared() / 2.)

    def update_qw(self):
        e, f, a, b, X, y = self.e, self.f, self.a, self.b, self.X, self.y
        self.diag = pd.DataFrame(np.diag([a/bk for bk in b]))
        self.Sigma = pd.DataFrame(np.linalg.inv(self.diag + ((e/f) * X.T.dot(X))))
        self.mu = self.Sigma.dot((e/f)*X.mul(y, axis=0).sum()).to_frame()

    def _sum_y_minus_xTw_squared(self):
        mu, Sigma, X, y = self.mu, self.Sigma, self.X, self.y
        term1 = X.dot(mu).subtract(y, axis=0) ** 2
        term2 = pd.DataFrame(np.diag(X.dot(Sigma).dot(X.T)))
        return (term1 + term2).sum()[0] # [0] to get value out of DF.

    # Performs much worse.
    # def _sum_y_minus_xTw_squared(self):
    #     '''Sum(y_i - x_i^Tw)^2'''
    #     Sigma, mu, X, y = self.Sigma, self.mu, self.X, self.y
    #     quadratic = np.diag(X.dot(self._ExxT()).dot(X.T))
    #     linear = mu.T.dot(X.mul(y, axis=0).T)
    #     return sum((y ** 2) - (2*linear) + quadratic)

    def _ExxT(self):
        '''E[xx^T] = Sigma - mu mu^T'''
        return self.Sigma + self.mu.dot(self.mu.T)

    def calcloglikelihood(self):
        e0, e, f0, f = self.e0, self.e, self.f0, self.f
        a0, a, b0, b = self.a0, self.a, self.b0, self.b
        N, d, Sigma, mu = self.N, self.d, self.Sigma, self.mu
        term1 = (e0*np.log(f0) - gammaln(e0)) - (e*np.log(f) - gammaln(e))
        term2 = (e0 - e)*(digamma(e) - np.log(f)) - (f0 - f)*(e/f)
        term3 = d*(a0*np.log(b0) - gammaln(a0)) - (a*sum([np.log(bk) for bk in b]) - d*gammaln(a))
        term4 = (a0 - a)*sum([(digamma(a) - np.log(bk)) for bk in b]) - (b0*sum([a/bk for bk in b])) + d*a
        term5 = sum([(digamma(a) - np.log(bk)) for bk in b]) + self._log_determinant(Sigma)
        term6 = - a*sum([el/bk for el, bk in zip(np.diag(self._ExxT()),b)])
        term7 = N*(digamma(e) - np.log(f))
        term8 = -(f/e)*self._sum_y_minus_xTw_squared()
        ll = term1 + term2 + term3 + term4 + (term5 + term6)/2. + (term7 + term8)/2
        self.loglikelihood.append(ll)



# HW 4: GMM
class GMMBaseModel(BaseModel):

    def _gen_mu(self):
        # TODO: Implement FFT
        X, k, d = self.X, self.k, self.d
        mu = pd.DataFrame(np.zeros((k, d)))
        points = list(X.index)
        random.shuffle(points)
        random.shuffle(points)
        random.shuffle(points)
        print points[:k]
        for j in xrange(k):
            mu.ix[j] = X.ix[points[j]] # To avoid drawing same point twice
        print mu
        return mu

    def _gen_Lambda(self):
        k, d = self.k, self.d
        idx = pd.MultiIndex.from_product([range(k), range(d)])
        Lambda = pd.DataFrame(np.random.randn(d*k, d), index=idx)
        for j in xrange(k):
            Lambda.ix[j] = np.eye(d)
        return Lambda

    def draw_clusters(self, means):
        X = self.X
        clusters = self.phi.idxmax(axis=1)

        # plt.scatter(self.X[0], self.X[1], c=clusters)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(X[0], X[1], c=clusters, marker="o", label='data')
        ax1.scatter(means[0], means[1], s=100, marker="x", label='means')
        plt.legend(loc='upper left');

        plt.show()

    def train(self):
        self.iterate(100)

    def iterate(self, n=1):
        for t in xrange(n):
            if not t % 10: print 'Iteration {}'.format(t)
            self._iterate()

class GMM_EM(GMMBaseModel):
    def __init__(self, X, k=3):
        self.X = X
        self.n = len(X)
        self.d = len(X.T)
        self.k = k
        self.phi = pd.DataFrame(np.full((self.n, self.k), 1./self.k)) # n x k
        self.pi = pd.Series(np.full(self.k, 1./self.k)) # 1 x k
        self.mu = self._gen_mu() # k x d
        self.Lambda = self._gen_Lambda() # d x d, store as cov, not precision.
        self.loglikelihood = []

    def _iterate(self):
        self.EStep()
        self.MStep()
        self.calcloglikelihood()

    def EStep(self):
        self.update_phi()

    def MStep(self):
        self.update_model_variables()

    def update_phi(self):
        '''Calculate probability for every pi_j / x_i combo, normalize'''
        X_probs = self.get_X_probs()
        scaled_probs = X_probs * self.pi
        self.phi = scaled_probs.div(scaled_probs.sum(axis=1), axis=0)

    def update_model_variables(self):
        phi, mu, X, Lambda = self.phi, self.mu, self.X, self.Lambda

        n = phi.sum()

        for j in mu.index:
            self.mu.ix[j] = (X.mul(phi[j], axis=0)).sum() / n[j]

        for j in Lambda.index.levels[0]:
            self.Lambda.ix[j] = self._update_Lambda(j)/n[j]

        self.pi = n / n.sum()

    def get_X_probs(self):
        X, mu, Lambda = self.X, self.mu, self.Lambda
        X_probs = pd.DataFrame(np.zeros((self.n, self.k)))
        for j in mu.index:
            X_probs[j] = multivariate_normal.pdf(X, mu.ix[j], Lambda.ix[j])
        return X_probs

    def _update_Lambda(self, j):
        tau = self.X - self.mu.ix[j]
        tau[2] = tau[0] * tau[1] # Assumes self.d = 2 !! TODO: Generalize
        tau[0] = tau[0]**2 # Assumes self.d = 2 !! TODO: Generalize
        tau[1] = tau[1]**2 # Assumes self.d = 2 !! TODO: Generalize
        tau = tau.mul(self.phi[j], axis=0)
        Tau = tau.sum()
        Lambda = np.eye(self.d)
        Lambda[0,0] = Tau[0]
        Lambda[0,1] = Tau[2]
        Lambda[1,0] = Tau[2]
        Lambda[1,1] = Tau[1]
        return Lambda

    def calcloglikelihood(self):
        '''sum_i(ln sum_j p(x))'''
        X_probs = self.get_X_probs()
        scaled_probs = X_probs * self.pi
        ll = np.log(scaled_probs.sum(axis=1)).sum()
        print 'Log likelihood:', ll
        self.loglikelihood.append(ll)

class GMM_VI(GMMBaseModel):
    def __init__(self, X, k=2, alpha=1, c=10):
        self.X = X
        self.n = len(X)
        self.d = len(X.T)
        self.k = k

        # Distribution Parameters
        self.c = c
        self.m_0 = self._gen_mu()
        self.Sigma_0 = self._gen_Lambda() * self.c
        self.B_0 = pd.DataFrame(self.X.cov() * (self.d / 10.))
        self.a_0 = pd.Series(np.full(self.k, self.d))
        self.alpha_0 = pd.Series(np.full(self.k, alpha))

        self.m = self.m_0.copy()
        self.Sigma = self.Sigma_0.copy()
        self.B = pd.concat([self.B_0 for _ in xrange(self.k)], keys=range(self.k))
        self.a = self.a_0.copy()
        self.alpha = self.alpha_0.copy()
        self.phi = pd.DataFrame(np.full((self.n, self.k), 1./self.k)) # n x k

        self.objective = []

    def iterate(self, n=1):
        for t in xrange(n):
            if not t % 10: print 'Iteration {}'.format(t)
            if not t % 10: print self.phi.sum()
            self._iterate()

    def _iterate(self):
        X, d, k, n = self.X, self.d, self.k, self.n
        a, a_0, B, B_0 = self.a, self.a_0, self.B, self.B_0
        Sigma, m, alpha = self.Sigma, self.m, self.alpha

        # Update q(c)
        exp = self.phi.copy()

        Binv = B.copy()
        for j in xrange(k):
            Binv.ix[j] = np.linalg.inv(B.ix[j])
        aBinv = Binv.mul(a, level=0, axis=0) # OK

        for j in xrange(k):
            diff = X - m.ix[j]
            t1 = (sum([digamma((1-l+a.ix[j]) / 2.) for l in xrange(1,d+1)])
                  + self._log_determinant(B.ix[j])) # OK
            t2 = pd.Series(np.diag(diff.dot(aBinv.ix[j]).dot(diff.T))) # OK!
            t3 = np.trace(aBinv.ix[j].dot(Sigma.ix[j])) # OK
            t4 = digamma(alpha.ix[j]) - digamma(alpha.sum()) # OK
            exp[j] = np.e ** ((t1 - t2 - t3)/2. + t4)

        self.phi = exp.div(exp.sum(axis=1), axis=0) # OK
        ###

        nj = self.phi.sum() # OK

        # Update q(pi)
        self.alpha = self.alpha_0 + nj # OK
        ###

        # Update q(mu)
        cI = np.eye(d) / self.c
        for j in xrange(k):
            self.Sigma.ix[j] = np.linalg.inv(cI + (aBinv.ix[j] * nj[j]))
            scalesum = X.mul(self.phi[j], axis=0).sum()
            self.m.ix[j] = self.Sigma.ix[j].dot(aBinv.ix[j].dot(scalesum))
        ###

        # Update q(Lambda)
        self.a = a_0 + nj # OK

        # BROKEN
        for j in xrange(k):
            sum1 = self._get_scaled_outer_products(j)
            sum2 = nj[j] * self.Sigma.ix[j]
            self.B.ix[j] = (B_0 + sum1 + sum2).values
        ###

        self.calc_objective(aBinv)

    def _get_scaled_outer_products(self, j):
        U = self.X - self.m.ix[j]
        tau = pd.DataFrame(np.zeros((self.n, self.d)))
        tau[0] = U[0] * U[0]
        tau[1] = U[0] * U[1]
        tau[2] = U[1] * U[0]
        tau[3] = U[1] * U[1]
        tau = tau.mul(self.phi[j], axis=0)
        Tau = tau.sum()
        matrix = pd.DataFrame(np.eye(self.d))
        matrix.iloc[0,0] = Tau[0]
        matrix.iloc[0,1] = Tau[1]
        matrix.iloc[1,0] = Tau[2]
        matrix.iloc[1,1] = Tau[3]
        return matrix


    def calc_objective(self, aBinv):
        d, X, k, c = self.d, self.X, self.k, self.c
        alpha, alpha_0 = self.alpha, self.alpha_0
        B, a, B_0, a_0 = self.B, self.a, self.B_0, self.a_0.ix[0]
        m, Sigma = self.m, self.Sigma
        phi = self.phi

        Binv = B.copy()
        for j in xrange(k):
            Binv.ix[j] = np.linalg.inv(B.ix[j])
        aBinv = Binv.mul(a, level=0, axis=0)
        ElnPi_0 = digamma(alpha_0) - digamma(alpha_0.sum())
        ElnPi = digamma(alpha) - digamma(alpha.sum())

        # Pi
        t1 = (gammaln(alpha.sum()) - gammaln(alpha).sum())
        t2 = -(gammaln(alpha_0.sum()) - gammaln(alpha_0).sum())
        t3 = (ElnPi).mul(alpha_0 - alpha, axis=0).sum()

        # Mu
        t4 = sum([self._log_determinant(Sigma.ix[j]) for j in m.index])*(d/5.)
        t5 = sum([np.trace(Sigma.ix[j] - np.outer(m.ix[j], m.ix[j])) for j in m.index])/(2.*c)

        # C
        t6 = phi.T.mul((ElnPi_0 - ElnPi), axis=0).sum().sum()

        # X
        t7 = 0
        for j in m.index:
            U = X - m.ix[j]
            t7 -= (pd.Series(np.diag(U.dot(aBinv.ix[j]).dot(U.T))) * phi[j]).sum()

        t8 = 0
        t9 = 0
        for j in m.index:
            phij = phi[j].sum()
            t8 += (np.trace(aBinv.ix[j] + Sigma.ix[j])) * (phij/2.)
            t9 += (self.ElnWishart(a.ix[j], B.ix[j])) * (phij/2.)

        # Lambda
        t10 = sum([(a_0 - a.ix[j])/2. * self.ElnWishart(a.ix[j], B.ix[j]) for j in m.index])
        t11 = sum([np.trace((B_0+B.ix[j])*aBinv.ix[j]) for j in m.index])/2.
        t12 = (a_0/2.)*self._log_determinant(B_0) * k
        t13 = sum([((a.ix[j]/2.)*self._log_determinant(B.ix[j])) for j in m.index])
        t14 = 0
        for j in m.index:
            t14 += sum([(gammaln((a_0+1-k)/2.) - gammaln((a.ix[j]+1-k)/2.)) for k in range(d)])

        obj = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 + t14
        self.objective.append(obj)

    def ElnWishart(self, a, B): # No constant terms
        return sum([digamma(a+1-j)/2 for j in xrange(1,self.d+1)]) + self._log_determinant(B)


class GMM_Gibbs(GMMBaseModel):
    '''
    Lambda ~ Wishart(a, B)
    Mu|Lambda ~ Normal(m, (cLambda)^{-1})
    '''
    def __init__(self, X, alpha=1, c=.1):
        self.X = X
        self.n = len(X)
        self.d = len(X.T)

        # Distribution Parameters
        self.alpha = alpha
        self.c = c
        self.a = self.d
        self.m = self.X.mean()
        self.B = self.c * self.d * self.X.cov()
        self.Binv = np.linalg.inv(self.B)

        self.Lambda = self._gen_Lambda()
        self.Mu = self._gen_Mu(self.Lambda.ix[0])
        self.clusters = pd.Series(np.zeros(self.n), dtype=np.dtype('int8'))
        self.numclusters = []
        self.topclusters = []
        self.clust_index = 0

    # Initialization
    def _gen_Lambda(self):
        idx = pd.MultiIndex.from_product([(0), range(self.d)])
        Lambda = self._draw_Lambda(self.a, self.B, self.c)
        return pd.DataFrame(Lambda, index=idx)

    def _gen_Mu(self, Lambda):
        return pd.DataFrame(self._draw_mu(self.m, Lambda)).T

    # Updates
    def _draw_Lambda(self, a, B, c):
        Binv = np.linalg.inv(B)
        Lambda =  wishart.rvs(a, Binv)
        return np.linalg.inv(c * Lambda)

    def _draw_mu(self, m, cov):
        return multivariate_normal.rvs(m, cov)

    def _get_prob(self, j, x):
        mu, Lambda = self.Mu.ix[j], self.Lambda.ix[j]
        return multivariate_normal.pdf(x, mu, Lambda)

    def _draw_discrete(self, probs):
        return np.random.choice(range(len(probs)), p=probs)

    def _px(self, x):
        c, a, d, m, B = self.c, self.a, self.d, self.m, self.B
        u = x - m

        term1 = (c/(np.pi*(1+c))) ** (d/2.)
        term2 = np.linalg.det(B + (c/(1+c))*np.outer(u, u)) ** (-(a+1)/2.)
        term3 = self.Binv ** (-a/2.)
        term4 = np.e ** sum([
            (gammaln((a+2-j)/2.) - gammaln(((a+1-j)/2.)))
            for j in xrange(1,d+1)
            ])

        return term1 * (term2 / term4) * term4

    def _update_theta(self, j):
        m, c, a, B = self.m, self.c, self.a, self.B
        c_idx = self.clusters[self.clusters == j].index
        cX = self.X.ix[c_idx]
        s = len(cX)
        xbar = cX.mean()
        U = cX - xbar
        sum_op = sum([np.outer(U.ix[i], U.ix[i]) for i in U.index])

        mj = (c*m + cX.sum())/(s+c)
        cj = s + c
        aj = a + s
        Bj = B + sum_op + ((s/(a*s+1)) * np.outer((xbar-m), (xbar-m)))

        Lambdaj = self._draw_Lambda(aj, Bj, cj)
        muj = self._draw_mu(mj, Lambdaj)

        # SUPER JANKY
        if j in self.Lambda.index.levels[0]:
            self.Lambda.ix[j] = Lambdaj
        else:
            idx = pd.MultiIndex.from_product([(int(j)), range(self.d)])
            Lambdaj = pd.DataFrame(Lambdaj, index=idx)
            self.Lambda = self.Lambda.append(Lambdaj)

        self.Mu.ix[j] = muj

    def train(self):
        self.iterate(15)

    def iterate(self, n=1):
        for t in xrange(n):
            # if not t % 10: print 'Iteration', t
            # if not t % 10: print 'Num clusters', len(self.Mu)
            print 'Iteration', t
            print 'Num clusters', self.clust_index
            self._iterate()

    def _iterate(self):
        nj = self.clusters.value_counts()
        probs = pd.DataFrame(np.zeros((self.n, len(nj))), columns=nj.index)
        for j in nj.index:
            probs[j] = self._get_prob(j, self.X)

        for i in self.X.index:
            x = self.X.ix[i]
            nj = self.clusters.value_counts()
            Kt = self.clust_index
            ci = self.clusters.ix[i]
            nj[ci] -= 1 # Remove x_i from counts
            phi = pd.Series(np.zeros(len(nj)), index=nj.index)

            denom = float(self.alpha + self.n - 1)

            for j in nj.index:
                try:
                    probx = probs.loc[i, j]
                except KeyError:
                    probs[j] = self._get_prob(j, self.X)
                phi[j] = (probx * nj[j]) / denom

            phi[Kt+1] = (self.alpha / denom) * self._px(x)

            phi = phi / phi.sum()
            assert np.abs(phi.sum() - 1) < .00001 # Normalized!

            cluster = self._draw_discrete(phi)
            self.clusters.ix[i] = cluster

            if cluster == Kt+1:
                print cluster, phi[cluster]
                self._update_theta(Kt+1)
                self.clust_index += 1
                probs[Kt+1] = self._get_prob(Kt+1, self.X)

            # Destroy cluster if empty
            if ci not in self.clusters.values:
                print 'Destroying', ci
                self.Mu.ix[ci] = float('NaN')
                self.Lambda.ix[ci] = float('NaN')

        nj = self.clusters.value_counts() # Latest values
        for j in nj.index:
            self._update_theta(j)

        self.numclusters.append(len(nj))
        nj.sort_values(inplace=True, ascending=False)
        top = nj[:6]
        top.index = range(6)
        self.topclusters.append(top)















