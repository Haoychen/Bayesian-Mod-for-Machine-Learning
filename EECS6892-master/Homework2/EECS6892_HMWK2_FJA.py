# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 10:21:31 2015
@author: franciscojavierarceo
"""
import scipy
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy import io, stats
import matplotlib.pyplot as plt

def plotNumber(q, x, rowval, data):
    print 'Plot and Summary for Observation ' + str(rowval)
    xtmp = np.array(x.iloc[rowval].T).reshape((15,1))
    x2 = np.dot(q,xtmp).reshape((28,28))
    plt.imshow(x2,interpolation='nearest')
    plt.savefig('Plot_Number'+str(rowval)+'.jpg')
    print data.iloc[rowval]

def plotNumberW(q, w, rowval, data):
    print 'Plot and Summary for Observation ' + str(rowval)
    predtmp = np.dot(q, w[rowval]).reshape((28,28)) 
    plt.imshow(predtmp ,interpolation='nearest')
    plt.savefig('Plot_W'+str(rowval)+'.jpg')
    print w[rowval]

def MyProbit(xtrn, ytrn,T, sigma, lambda_val):
    n,d = xtrn.shape
    # Initializing the weights to zero
    wt = np.zeros((d,1))
    phi = np.zeros((n,1))
    # useful for indexing
    flt1 = np.where(ytrn==1)
    flt0 = np.where(ytrn==0)
    lnpyw_x=[]
    wtmp = []
    xxt = np.zeros((d,d))
    for i in xrange(n):
        xxt = xxt+ np.dot(xtrn.iloc[i].reshape(15,1),xtrn.iloc[i].reshape(1,15))
        
    xxt = np.dot(xtrn.T,xtrn)
    for t in xrange(T):
        xw = np.dot(xtrn,wt).reshape((n,1))
        xwbysigma = np.divide(xw, sigma)
        # E-step
        phi[flt1] = xw[flt1] + (sigma * norm.pdf( -xwbysigma[flt1]  ) / (1.0-norm.cdf(- xwbysigma[flt1] ) ) )
        phi[flt0] = xw[flt0] + (sigma * -norm.pdf( -xwbysigma[flt0] ) / (norm.cdf(- xwbysigma[flt0]) ) )
        # M-step        
        wt = np.dot(np.linalg.inv( lambda_val* np.diag(np.ones(d)) + np.divide(xxt,sigma**2.0)), 
                    np.sum(np.divide(np.multiply(xtrn,phi),sigma**2.0),axis=0))
        wtmp.append(wt)
        # Calculating lnp(y,w_t|X)
        lnpyw_x.append( float((d/2.0) * np.log(lambda_val / (2.0*np.pi)) - np.multiply((lambda_val/2.0),np.dot(wt.T,wt))+
                np.dot(ytrn.T,np.log( norm.cdf(xwbysigma) )) + np.dot( (1.0-ytrn).T,np.log( 1.0 - norm.cdf(xwbysigma)))))
        if (t+1) % 10 == 0:
            print "Iteration " + str(t+1)            
    return wt, lnpyw_x, wtmp

def ProbPred(xtst, wt):
    n = xtst.shape[0]
    yprob = np.dot(xtst,wt).reshape((n,1))
    ypred = np.array(mysign(yprob)).reshape((n),1)
    return yprob, ypred

def mysign(x):
    # Comprehension loops
    return [0 if i <= 0 else 1 for i in x]

def ConfusionMatrix(pred, actual):
    return pd.crosstab(pred,actual)

def PrintAccuracy(pred, actual):
    perf = pd.crosstab(pred,actual).values
    print( "Model accuracy is (" + str(perf[0][0] + perf[1][1]) +"/" + str(len(pred)) + ") = " +
         str( (perf[0][0] + perf[1][1])/float(len(pred))) )

if __name__ == '__main__':
    path = '/Users/franciscojavierarceo/GitHub/EECS6892/Homework1/hw1_data_csv/'
    # Training data
    xtrn = pd.read_csv(path+'Xtrain.csv', header=None)
    ytrn = pd.read_csv(path+'ytrain.csv', header=None)
    # Reading test data
    xtst = pd.read_csv(path+'Xtest.csv', header=None)
    ytst = pd.read_csv(path+'ytest.csv', header=None)
    # Need Q to project data back into an image
    Q = pd.read_csv(path+"Q.csv",header=None)

    # Running model
    wt, lnpyw_x, wtmp = MyProbit(xtrn,ytrn,100,1.5,1.0)
    yprob, ypred = ProbPred(xtst,wt)
    preds = pd.DataFrame( norm.cdf(yprob))
    preds.columns = ['PredProb']    
    preds['Predicted'] = ypred
    preds['Actual'] = ytst
    preds['Incorrect'] = np.absolute(np.array(preds['Predicted'],dtype=int) -preds['Actual'])
    # The most ambiguous prediction is defined for those closest to 0.5 in absolute value
    preds['Ambig'] =  np.absolute(preds['PredProb']-0.5000 )
    top3 = preds['Ambig'].argsort()[0:3].values
    
    # Note, y = 0 means 4 and y = 1 means 9
    print(ConfusionMatrix(preds['Actual'],preds['Predicted']))

    PrintAccuracy(preds['Actual'],preds['Predicted'])
    incorrect = preds.index[preds['Incorrect']==1][0:3]

    print("Printing 3 missclassified")
    for row_data in incorrect:
        plotNumber(Q, xtrn, row_data, preds)
    
    print("Printing 3 most ambiguous")
    for row_data in top3:
        plotNumber(Q, xtrn, row_data, preds)
    
    print("Printing the Weights at various iterations")
    for i in [1, 5, 10, 25, 50, 100]:
        plotNumberW(Q, wtmp, (i-1), preds)

    plt.figure(figsize=(12,8))
    plt.plot(lnpyw_x)
    plt.title('Likelihood distribution of y and w')
    plt.ylabel('Likelihood Value')
    plt.xlabel('Iteration')
    plt.savefig('LLK.jpg')

#------------------------------------
# End 
#------------------------------------