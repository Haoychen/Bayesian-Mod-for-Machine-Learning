# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:49:31 2015
@author: franciscojavierarceo
"""
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import scipy

def plotNumber(q, x, rowval, data):
    print 'Plot and Summary for Observation ' + str(rowval)
    xtmp = np.array(x.iloc[rowval].T).reshape((15, 1))
    x2 = np.dot(q, xtmp).reshape((28,28))
    plt.imshow(x2, interpolation='nearest')
    print data.iloc[rowval]
    plt.savefig('Plot_Row%i.jpg'  % rowval)

def PosteriorX(a, b, xtrn, xtst, ytrn, val):
    sdx = np.std(np.hstack((xtrn,xtst)))    # This cleans it up a bit
    tmpx = xtrn[np.where(ytrn==val)[0]]     # X given Y = j 
    xtrn_yj= tmpx / sdx                     # Scaling the xs
    xbar = np.mean(xtrn_yj)                 # Conditional mean
    n = float(xtrn_yj.shape[0])             # Number of observations
    v = 2.0*b + n
    mu_x = n*xbar/float((1/float(a)+n))
    scale = 2.0 * float((a*n+a+1)) / float((v+a*n*v))
    tout = (-(v+1.0)/2.0)*np.log(1.0+((xtst/sdx -mu_x)**2.0) / (scale*v))
    return tout.reshape((len(xtst),1))

def PrintAccuracy(pred, actual):
    perf = pd.crosstab(pred,actual).values
    print "Model accuracy is " + str( (perf[0][0] + perf[1][1])/float(len(pred)))
    print str(perf[0][0] + perf[1][1]) +"/" + str(len(pred))

if __name__ == '__main__':
    # Specify the path of the data
    path = '/Users/franciscojavierarceo/GitHub/EECS6892/Homework1/hw1_data_csv/'
    # Training data
    xtrn =pd.read_csv(path+'Xtrain.csv', header=None)
    ytrn =pd.read_csv(path+'ytrain.csv', header=None)
    # Reading test data
    xtst =pd.read_csv(path+'Xtest.csv', header=None)
    ytst =pd.read_csv(path+'ytest.csv', header=None)
    # Need Q to project data back into an image
    Q = pd.read_csv(path+"Q.csv", header=None)

    # Setting Hyper Parameters    
    a,b,c,d,e,f = [1.0]*6
    # Class priors
    post_y1 = float((e + sum(ytrn==1)) / (len(ytrn)+e+f))
    post_y0 = float((f + sum(ytrn==0)) / (len(ytrn)+e+f))
    # Initializing the predictions to 0 -- so that I can just add them
    pred_y1 = np.zeros((len(ytst),1))
    pred_y0 = np.zeros((len(ytst),1))
    k = xtst.shape[1]

    for i in range(0, k):
        print('Evaluating feature %i' % i)
        post_y1x1 = PosteriorX(1, 1, xtrn[i], xtst[i], ytrn, 1) * post_y1
        post_y0x0 = PosteriorX(1, 1, xtrn[i], xtst[i], ytrn, 0) * post_y0
        pred_y1 = post_y1x1 + pred_y1
        pred_y0 = post_y0x0 + pred_y0
        if (i== k-1):
            print "End"    

    preds = pd.DataFrame(pred_y0)
    preds['1'] = pred_y1
    preds['Predicted'] = preds.idxmax(1)
    preds['Actual'] = ytst

    tmp = preds.ix[:,0]+preds.ix[:,1] # Normalizing to get the probabilities
    preds['Prob_y0'] = 1.0- preds.ix[:,0]/tmp   # Have to do 1-x
    preds['Prob_y1'] = 1.0- preds.ix[:,1]/tmp
    preds['Incorrect'] = np.absolute(np.array(preds['Predicted'],dtype=int) -preds['Actual'])
    preds['Ambig'] =  np.absolute(preds['Prob_y1']-0.5000 )

    # Most ambiguous predictions are those with absolute value closest to 0.5    
    x, y = preds.index, preds['Ambig']
    top3 = preds['Ambig'].argsort()[0:3]

    # Note, y = 0 means 4 and y = 1 means 9
    print pd.crosstab(preds['Actual'],preds['Predicted'])
    PrintAccuracy(preds['Actual'],preds['Predicted'])

    incorrect= preds.index[preds['Incorrect']==1][0:3]
    
    print('Three misclassified observations are:')
    for i in incorrect:
        plotNumber(Q, xtrn, i ,preds)
    
    print("Three most ambiguous predictions are " + str(top3.values))

    for t3 in top3:
        plotNumber(Q,xtrn,top3[0],preds)

#------------------------------------
# End 
#------------------------------------