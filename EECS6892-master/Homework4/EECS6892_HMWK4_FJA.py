# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:40:56 2015

@author: franciscojavierarceo
"""
import scipy
from pylab import *
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import io, stats
import matplotlib.pyplot as plt

# from scipy.stats import multivariate_normal.pdf as mvnorm
# from np.linalg import inv, det
# from np.trace import trace
# from np.log import ln
# from scipy.random import random as ran
# from np.random.normal import rnorm
# from scipy.special import psi
# from np.diag import diag
# from stats.dirichlet import entropy
# from scipy.gammaln import gammaln
# from stats.multivariate_normal import rvs as mvnorm_sample
# from stats.wishart import rvs as wishart_sample 
# from stats.dirichlet import rvs as dirichlet_sample

#------------------------------------------------------------------------------
# 1.---------------------------------------------------------------------------
# We are given observations X = {x1, . . . , xn} where generated from a Gaussian mixture model of the form
#           xi | ci ∼ N ormal(μci , Λci ),  ci ∼ Discrete(π)
# In this problem, you will implement the EM algorithm for learning maximum 
# likelihood values of π and each (μj,Λj) for j = 1,...,K. 
#------------------------------------------------------------------------------
# a. Implement the EM-GMM algorithm and run it for 100 iterations on the data 
#        provided for K = 2,4,8,10.    

# b. For each K, plot the log likelihood over the 100 iterations. 
#       What pattern do you observe and why might this not be the best way 
#       to do model selection?
 
# c. For the final iteration of each model, plot the data and indicate the 
#       most probable cluster of each observation according to q(ci) by a 
#       cluster-specific symbol. What do you notice about these plots as 
#       a function of K?
#------------------------------------------------------------------------------

mvnorm = scipy.stats.multivariate_normal.pdf
inv = np.linalg.inv 
det = np.linalg.det
tr = np.trace
ln = np.log
ran = scipy.random.random
rnorm = np.random.normal
psi = scipy.special.psi
diag = np.diag
dentropy = stats.dirichlet.entropy
gammaln = scipy.special.gammaln
mvnorm_sample = stats.multivariate_normal.rvs
wishart_sample = stats.wishart.rvs
dirichlet_sample= stats.dirichlet.rvs

def EM_GMM(xtrn, k, T):
    n, d = xtrn.shape
    pi_j = np.ones(k)
    phi_i = ran((n,k))      
    mu_j = asarray([ran(d) for i in xrange(k)])
    lambda_j = asarray([np.diag(ran(d)) for i in xrange(k)])
    llk_t = []
    for t in range(T):
        # E-Step
        for j in range(k):
            for i, xi in enumerate(xtrn.values):
                phi_i[i,j] = pi_j[j] * mvnorm(xi, mu_j[j], inv(lambda_j[j])) 
                
        # Summing over to normalize
        phi_i = phi_i / np.sum(phi_i,axis=1).reshape((n,1)) 
        # M-Step
        n_j = np.sum(phi_i,axis=0)
        for j in range(k):
            mu_j[j] = (1.0/n_j[j]) * np.sum(phi_i[:,j].reshape((n,1)) * xtrn.values ,axis=0)
                    
        for j in range(k):
            phixmuxT = np.zeros((d,d))    
            for i, xi in enumerate(xtrn.values):
                xminusmu = (xi- mu_j[j]).reshape((d,1))
                phixmuxT += phi_i[i][j] * xminusmu.dot(xminusmu.T)
            lambda_j[j]= inv( (1/n_j[j]) *phixmuxT )

        pi_j = n_j / np.sum(n_j)
        lnp=0
        for i, xi in enumerate(xtrn.values):
            px=0
            for j in range(k):
                px+=pi_j[j]*mvnorm(xi, mean=mu_j[j], cov=inv(lambda_j[j]))
            lnp+=ln(px)        
        llk_t.append(lnp)
        if ( (t+1) % 10)== 0:
            print "Iteration "+str(t+1)+" Complete"
    return phi_i, mu_j, pi_j, lambda_j, n_j,llk_t

def llkplot(dlist, k, inpth):
    myplot = plt.figure(figsize=(12,8))
    plt.plot(dlist[k][5])
    plt.title('EM for Gaussian Mixture Model with K='+k[5:])
    plt.xlabel('Iterations')
    plt.ylabel('Log-likelihood')
    plt.grid()
    myplot.savefig(inpth+'HMWK4_Q1b_'+k+'.png')
    
def em_scatplot(dlist, xtrn, k, inpth):
    cluster = [i.argmax() for i in dlist[k][0]]
    myplot = plt.figure(figsize=(12,8))
    plt.scatter(xtrn[0].values,xtrn[1].values,c=cluster,s=40)
    plt.title('Scatter plot for Gaussian Mixture Model with K='+k[5:])
    plt.grid()
    plt.xlabel("X1")
    plt.ylabel("X2")
    myplot.savefig(inpth+'HMWK4_Q1c_'+k+'.png')
#------------------------------------------------------------------------------
# 2.---------------------------------------------------------------------------
# In this problem, you will implement a variational inference algorithm for 
# approximating the posterior distribution of the GMM variables. We therefore 
# require prior distributions on these variables. For this problem, we use
#       π ∼ Dirichlet(α), μj ∼ Normal(0,cI), Λj ∼ Wishart(a,B)
# For this problem, set α = 1, set m to be the empirical mean of the data, 
# c = 10, a = d and B = d A where A is the empirical covariance of the data. 
# Approximate the posterior distribution 10 of these variables with q 
# distributions factorized on π, and each μj, Λj and ci as discussed in class.
#------------------------------------------------------------------------------
# a. Implement the variational inference algorithm discussed in class and in 
#       the notes for K = 2, 4, 10, 25 and 100 iterations each.

# b. For each K, plot the variational objective function over the 100 
#       iterations. What pattern do you observe?

# c. For the final iteration of each model, plot the data and indicate the 
#       most probable cluster of each observation according to q(ci) by a 
#       cluster-specific symbol. What do you notice about these plots as 
#       a function of K? 
#------------------------------------------------------------------------------
def samp(xtrn):
    s1 = scipy.random.choice(xtrn[0],1)
    s2 = scipy.random.choice(xtrn[1],1)
    return np.array((s1[0],s2[0]))
    
def VI_GMM(xtrn, K, T, seedval=12):    
    np.random.seed(seedval)
    n, d = xtrn.shape
    alpha = 1.0
    c = 10.0
    a = float(d) 
    B = (d /10.0)* np.cov(xtrn.T)
    n_j = np.ones(K)
    a_j = ran(K) * 10
    sigma_j = [ran((d,d)) for i in range(K)]
    m_j = [ samp(xtrn) for i in range(K)]
    alpha_j = ran(K) * 10
    B_j = [diag(diag(B)) * ran(1) for i in range(K)]
    t1_j = np.zeros((n,K))
    t2_j = np.zeros((n,K))
    t3_j = np.zeros((n,K))
    t4_j = np.zeros((n,K))
    phi_ij = np.ones((n,K)) * (1.0/K)
    cI = diag([float(1/c)]*d)
    vof = np.zeros(T)
    for t in range(T):
        for i, xi in enumerate(xtrn.values):
            for j in range(K): # Calculating the terms for Phi, (t1,t2,t3, and t4)
                xminusm = (xi - m_j[j]).reshape((d,1))
                t1_j[i,j] = np.sum([psi( (float(k) + a_j[j])/2.0 ) for k in range(d)]) - ln(det(B_j[j]))
                t2_j[i,j] = xminusm.T.dot( ( a_j[j]* inv(B_j[j]) ) ).dot(xminusm)
                t3_j[i,j] = tr( (a_j[j]*inv(B_j[j])).dot(sigma_j[j]) )
                t4_j[i,j] = psi( alpha_j[j] ) - psi( np.sum(alpha_j) )
        # Calculating the numerator and denominator for each phi_j        
        phi_num = np.zeros((n,K))
        phi_denom = np.zeros((n,1))
        for j in range(K):
            phi_num[:,j] = np.exp( 0.5*t1_j[:,j] - 0.5*t2_j[:,j]- 0.5*t3_j[:,j] + t4_j[:,j])    
            phi_denom+= phi_num[:,j].reshape((n,1))
        # Updating each phi_i for each cluster
        phi_ij = np.divide(phi_num ,phi_denom)
        # Updating n_j 
        for j in range(K):
            n_j[j] = np.sum(phi_ij[:,j])
            # Updating q(pi)
            alpha_j[j] = alpha + n_j[j]
            # Updating q(Lj)
            sigma_j[j] = inv( cI + (n_j[j]*a_j[j]) * inv(B_j[j]))   
            m_j[j] = sigma_j[j].dot( a_j[j]* inv(B_j[j]).dot( np.sum(phi_ij[:,j].reshape(n,1) * xtrn))  )
        # Updating q(Lj)
        for j in range(K):
            bsum = 0.0
            for i, xi in enumerate(xtrn.values):
                xminusm = (xi-m_j[j]).reshape((d,1))
                bsum+= phi_ij[i,j] * (xminusm.dot(xminusm.T) + sigma_j[j])
            a_j[j] = a + n_j[j]
            B_j[j] = B+bsum
        Eln_qpi = dentropy(alpha_j)
        Eln_qmu  = [stats.multivariate_normal.entropy(m_j[j],sigma_j[j]) for j in range(K)]
        Eln_qLambda= [stats.wishart.entropy(a_j[j], inv(B_j[j])) for j in range(K)]
        Eln_ppi=(alpha-1.0)*(psi(alpha_j)-psi(np.sum(alpha_j)))
        Eln_pmu=-0.5/c*np.asarray([np.trace(sigma_j[j])+m_j[j].dot(m_j[j]) for j in range(K)])
        Eln_Lambda=-np.log(det(B_j))+np.sum(psi((a_j.reshape(K,1)+1.0-np.arange(1,d+1))/2.0),1)
        Eln_pi=psi(alpha_j)-psi(np.sum(alpha_j))
        E_xmuLambda=np.zeros((n,K))
        for i, xi in enumerate(xtrn.values):
             for j in range(K):
                 ximinusmj=(xi-m_j[j]).reshape(d,1)
                 E_xmuLambda[i,j]=ximinusmj.T.dot(a_j[j]*inv(B_j[j])).dot(ximinusmj)+np.trace(a_j[j]*inv(B_j[j]).dot(sigma_j[j]))
        Eln_pLambda=np.multiply(a_j-d-1.0,Eln_Lambda)/2.0-0.5*np.asarray([np.trace(B_j[j].dot(a_j[j]*inv(B_j[j]))) for j in range(K)])
        Eln_pxc=np.sum(np.multiply(phi_ij,0.5*Eln_Lambda+Eln_pi-0.5*E_xmuLambda),0)
        L=sum(Eln_ppi)+sum(Eln_pmu)+sum(Eln_pxc)+sum(Eln_pLambda)-Eln_qpi-sum(Eln_qmu)-sum(Eln_qLambda)
        vof[t]=L
    return phi_ij, m_j, sigma_j, alpha_j, a_j, B_j, vof

def vi_scatplot(phi_ij, xtrn, k, inpth):
    cluster = [i.argmax() for i in phi_ij]
    myplot = plt.figure(figsize=(12,8))
    plt.scatter(xtrn[0].values,xtrn[1].values,c=cluster,s=100)
    plt.title('Scatter plot for VI Gaussian Mixture Model with K='+str(k))
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid()
    myplot.savefig(inpth+'HMWK4_Q2a_'+str(k)+'.png')

def vof_plot(vof, inpth, k):
    myplot = plt.figure(figsize=(12,8))
    plt.plot(vof)
    plt.title('VI Objective Function for VI Gaussian Mixture Model with K='+str(k))
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Variational Objective Function')
    myplot.savefig(inpth+'HMWK4_Q2b_'+str(k)+'.png')

#------------------------------------------------------------------------------
# 3.---------------------------------------------------------------------------
# In this problem, you will implement a Bayesian nonparametric sampler for a 
# marginalized version of the GMM. In contrast to Problem 2, in this problem 
# we will use a joint prior on (μj,Λj). This is done for computational 
# convenience in calculating the marginal distribution of the data. 
# Specifically, we use the prior distribution
#       μj |Λj ∼ Normal(m,(cΛ)−1), Λj ∼ Wishart(a,B) 
# as well as the limit of the prior π ∼ Dirichlet(α/K,...,α/K) as K → ∞.
# In this problem you will implement the marginal sampler where π is 
# integrated out. For this problem, set m to be the empirical mean of the 
# data, c = 1/10, a = d and B = c · d · A where A is the empirical covariance 
# of the data. For the “cluster innovation parameter” set α = 1.
#------------------------------------------------------------------------------
# a. Implement the above-mentioned Gibbs sampling algorithm discussed in 
#       class and described in the notes. Run your algorithm on the data 
#       provided for 500 iterations.

# b. Plot the number of observations per cluster as a function of iteration 
#       for the six most probable clusters. These should be shown as lines that 
#       never cross; for example the ith value of the “second” line will be 
#       the number of observations in the second largest cluster after 
#       completing the ith iteration. If there are fewer than six clusters 
#       then set the remaining values to zero.

# c. Plot of the total number of clusters that contain data as a function of 
#       iteration.
#------------------------------------------------------------------------------
def update_posterior(xtrn, a, B, c, m):
    s, d = xtrn.shape
    xbar = np.mean(xtrn,0)
    sum_xbarxT = 0
    for xi in xtrn.values:
        xminusxbar = (xi-xbar).reshape(d,1)
        sum_xbarxT+= xminusxbar.dot(xminusxbar.T)
    xmxbar = (xbar-m).dot((xbar-m).T)
    m_j = (c/(s+c))*m+ (1/(s+c))*np.sum(xtrn,axis=0)
    c_j = s + c 
    a_j = s + a
    B_j = B + sum_xbarxT+ (s/(a*s+1.0))*xmxbar
    lambda_val = wishart_sample(a_j,inv(B_j))
    mu = mvnorm_sample(mean=m_j,cov=inv(c_j*lambda_val))
    return mu, lambda_val

def calculate_px(xi, d, a, B, c, m):
    xm = (xi-m.reshape((1,d))).reshape((d,1))
    xm_mx= xm.dot(xm.T)
    t1 = (c/(pi*(1.0+c)) )**(d/2)  
    t2 = det(B+ (c/(1.0+c))*xm_mx )**(-(a+1.0)/2.0) / det(B)**(-a/2.0)
    t3 = np.exp(np.sum([gammaln((a+1.0)/2.0- float(j)/2.0 )-gammaln(a/2.0-float(j)/2.0) for j in range(int(d))]))
    return t1 * t2 * t3

def GMM_Gibbs_Sampler(xtrn, T):
    n,d = xtrn.shape
    c = (1.0/10.0)
    a = float(d)
    B = c*float(d)* (np.cov(xtrn.T))
    m =np.mean(xtrn,axis=0)
    mu = [np.zeros(n) for i in range(n)]
    lambda_val = [np.zeros((n,n)) for i in range(n)]
    alpha =1.0
    c_i = np.zeros(n)
    phi_ij = np.zeros((n,n))
    phi_ij[:,0]=1.0
    mu[0], lambda_val[0] = update_posterior(xtrn,a,B,c,m)
    obspercluster = range(T)
    for t in range(T):
        phi_ij = np.zeros((n,n))
        for i, xi in enumerate(xtrn.values):
            ntmp=np.asarray([len(np.where(c_i==z)[0]) if z!=c_i[i] else len(np.where(c_i==z)[0])-1 for z in range(n)])
            n_j=np.where(ntmp>0)[0]
            for j in n_j:
                phi_ij[i,j] = mvnorm(xi,mu[j],inv(lambda_val[j])) * ntmp[j]/(alpha+n-1)
            jprime=int(max(set(c_i))+1)
            p_xi = calculate_px(xi,d,a,B,c,m)
            phi_ij[i,jprime] = alpha / (alpha + n - 1) * p_xi
            phi_ij[i] = phi_ij[i] / np.sum(phi_ij[i])
            idx = np.where(phi_ij[i])[0]
            dirch_sample = dirichlet_sample(phi_ij[i][phi_ij[i]>0])
            c_i[i] = idx[np.argmax(dirch_sample)]
            if c_i[i]==jprime:
                mu[jprime], lambda_val[jprime] = update_posterior(xtrn[c_i==jprime],a,B,c,m)
        for k in set(c_i):
            mu[int(k)], lambda_val[int(k)] = update_posterior(xtrn[c_i==k],a,B,c,m)
        obspercluster[t] = [np.sum(c_i==i) for i in set(c_i)]
        if( (t+1)%10==0):
            print "There are", len(set(c_i)),"clusters at iteration",(t+1),'distributed as',obspercluster[t]
    for t in obspercluster:
        t.sort(reverse=True)
    return c_i, phi_ij, mu, lambda_val, obspercluster

def plot_ClusterCount(cluster_counts_t, inpth):
    maxval =[]
    for i in cluster_counts_t:
        maxval= max(np.max(len(i)),maxval)
    plotdf = np.zeros((len(cluster_counts_t),maxval))
    for i,cvals in enumerate(cluster_counts_t):
        for j, cnums in enumerate(cvals):
            plotdf[i,j] = cnums
    myplot = plt.figure(figsize=(12,8))
    plt.plot(plotdf)
    plt.title('Count of Obs per Cluster for Gibbs-Sampler')
    plt.xlabel('Iteration')    
    plt.ylabel('Number of Observations per cluster')
    plt.grid()
    myplot.savefig(inpth+'HMWK4_Q3b.png')
    xtrn =pd.read_csv(inpth+'data.txt',header=None)

def plot_TotalClusters(cluster_counts_t, inpth):
    pltdf= [len(i) for i in cluster_counts_t]
    myplot = plt.figure(figsize=(12,8))
    plt.plot(pltdf)
    plt.title('Total Number of Clusters for Gibbs-Sampler')
    plt.xlabel('Iteration')    
    plt.ylabel('Total Number of clusters')
    plt.grid()
    myplot.savefig(inpth+'HMWK4_Q3c.png')

if __name__ == '__main__':
    # inpth = '/Users/franciscojavierarceo/GitHub/EECS6892/Homework4/'
    xtrn =pd.read_csv('data.txt',header=None)

    # print '*'*60,'\n','Running Expectation Maximation','\n','*'*60
    # emgmms= {}
    # for k in (2,4,8,10):
    #     emgmm = "emgmm%d" % k
    #     emgmms[emgmm]= EM_GMM(xtrn,k,100)
    #     llkplot(emgmms,emgmm,inpth)
    #     em_scatplot(emgmms,xtrn,emgmm,inpth)

    # print '*'*60,'\n','Running Variational Inference','\n','*'*60
    # vigmms= {}
    # for k in (2,4,10,25):
    #     vigmm = "vigmm%d" % k
    #     vigmms[vigmm]= VI_GMM(xtrn,k,100,1)
    #     vi_scatplot(vigmms[vigmm][0],xtrn,k,inpth)
    #     vof_plot(vigmms[vigmm][6],inpth,k)

    VI_GMM(xtrn, K=2, T=100)

    # print '*'*60,'\n','Running Gibbs-Sampler','\n','*'*60
    # gsgmm = GMM_Gibbs_Sampler(xtrn,500)
    # plot_ClusterCount(gsgmm[4],inpth)
    # plot_TotalClusters(gsgmm[4],inpth)
    # print '*'*60,'\n','END','\n','*'*60
#------------------------------------
# End 
#------------------------------------