# #Allison Fenichel 
# #Homework4
# #EECS6892

import scipy
from scipy import io, stats, special
from scipy.special import gamma, digamma, psi, polygamma, gammaln
import pandas as pd
import numpy as np
from numpy.linalg import inv, pinv, det
import math
from matplotlib import colors
import matplotlib.pyplot as plt
import itertools
import sys
import os



def EM_GMM(K):
	np.random.seed(6) #3
	dat=io.loadmat('data.mat')
	X=dat['X'].T
	N,d=X.shape
	T=100
	pi=np.ones((K))
	mu=scipy.random.random((K,d))
	Lamb=np.zeros((d,K,d))
	for j in range(K):
		Lamb[:,j,:]=np.diag(scipy.random.random(d))
	
	phi=scipy.random.random((N,K))
	n=np.zeros((K))	
	lnp=np.zeros(T)	
	for t in range(T):
		phi=e_step(K, X, mu, pi, Lamb, phi,n)
		n,mu,Lamb,pi=m_step(phi, K, X, mu, pi, Lamb, n)
		lnp[t]=likelihood(phi, K, X, mu, pi, Lamb, n)
	return phi,n,mu,Lamb,pi,lnp,X

def e_step(K, X, mu, pi, Lamb, phi, n):
	N,d=X.shape
	for i in range(N):
		denom=0
		for k in range(K):
			denom=denom+pi[k]*stats.multivariate_normal.pdf(X[i,:], mean=mu[k,:], cov=inv(Lamb[:,k,:]))
		for j in range(K):
			phi[i,j]=pi[j]*stats.multivariate_normal.pdf(X[i,:], mean=mu[j,:], cov=inv(Lamb[:,j,:]))/denom
	return phi		

def m_step(phi, K, X, mu, pi, Lamb, n):
	N,d=X.shape
	for j in range(K):
		n[j]=np.sum(phi[:,j],0)
		mu[j,:]=1.0/n[j]*np.sum(np.multiply(phi[:,j].reshape(N,1),X),0)
		s=0
		for i in range(N):
			xminusmu=X[i,:]-mu[j,:]
			xminusmu=xminusmu.reshape(d,1)
			s=s+phi[i,j]*np.dot(xminusmu,xminusmu.T)	
		Lamb[:,j,:]=inv(1.0/n[j]*s)
		pi[j]=n[j]/N
	return n,mu,Lamb,pi	

def likelihood(phi, K, X, mu, pi, Lamb, n):
	lnp=0
	N,d=X.shape
	for i in range(N):
		px=0
		for k in range(K):
			px=px+pi[k]*stats.multivariate_normal.pdf(X[i,:], mean=mu[k,:], cov=inv(Lamb[:,k,:]))
		lnp=lnp+np.log(px)
	return lnp


def VI(K):# mu,Lamb, pi, n):
	K=int(K)
	if K<10:
		np.random.seed(50)
	else:
		np.random.seed(6)
	dat=io.loadmat('data.mat')
	X=dat['X'].T
	N,d=X.shape
	T=100
	L=np.zeros(T)
	a_c=float(d)
	A=np.cov(X.T)
	B_c=d/10.0*A
	alpha_c=1.0
	c=10.0

	m=np.random.rand(K,d)
	alpha=np.random.rand(K)*N
	a=np.random.rand(K)*K**2 
	sigma=np.repeat([1/c*np.eye(d)],K,axis=0)
	B=np.asarray([(k+1)*B_c for k in range(K)])
	
	
	for t in range(T):
		phi=update_qc(K, X, a, B, m, alpha, sigma)
		n=np.sum(phi,0)
		alpha=alpha_c+n
		sigma=np.asarray([inv(1.0/c*np.eye(d)+n[k]*a[k]*inv(B[k])) for k in range(K)])
		for j in range(K):
			m[j]=sigma[j].dot(a[j]*inv(B[j])).dot(np.sum(np.multiply(phi[:,j].reshape(N,1),X),0))
			s=0
			for i, xi in enumerate(X):
				xminusm=xi-m[j]
				xminusm=xminusm.reshape(d,1)
				s=s+phi[i,j]*(np.dot(xminusm,xminusm.T)+sigma[j])
			B[j]=B_c+s
		a=a_c+n	
		L[t]=objective_function(K, X, alpha, a, B, m, sigma, phi, alpha_c, c)
	return phi, alpha, sigma, m, n, a, B, L

def update_qc(K, X, a, B, m, alpha, sigma):
	N,d=X.shape
	t1=np.sum(psi((a.reshape(1,K)-np.arange(d).reshape(d,1))/2.0)-np.log(det(B)),0)
	t3=np.asarray([np.trace(np.dot(a[k]*inv(B[k]), sigma[k])) for k in range(K)])
	t4=psi(alpha)-psi(np.sum(alpha))
	t2=np.zeros((N,K))
	for k in range(K):
		for i, xi in enumerate(X):
			xminusm=(xi-m[k]).reshape(d,1)
			t2[i,k]=(xminusm.T).dot(a[k]*inv(B[k])).dot(xminusm)
	numerator=np.exp(0.5*t1.reshape(1,K)-0.5*t2.reshape(N,K)-0.5*t3.reshape(1,K)+t4.reshape(1,K))
	denom=np.sum(numerator,1)
	phi=np.divide(numerator,denom.reshape(N,1))
	return phi		




def objective_function(K, X, alpha, a, B, m, sigma, phi, alpha_c, c):
	N,d=X.shape
	Eln_qpi=stats.dirichlet.entropy(alpha)
	Eln_qmu=[stats.multivariate_normal.entropy(m[j],sigma[j]) for j in range(K)]
	Eln_qLambda=[stats.wishart.entropy(a[j], inv(B[j])) for j in range(K)]
	Eln_ppi=(alpha_c-1.0)*(psi(alpha)-psi(np.sum(alpha)))
	Eln_pmu=-0.5/c*np.asarray([np.trace(sigma[j])+m[j].dot(m[j]) for j in range(K)])
	Eln_Lambda=-np.log(det(B))+np.sum(psi((a.reshape(K,1)+1.0-np.arange(1,d+1))/2.0),1)
	Eln_pi=psi(alpha)-psi(np.sum(alpha))
	E_xmuLambda=np.zeros((N,K))
	for i, xi in enumerate(X):
		for j in range(K):
			ximinusmj=(xi-m[j]).reshape(d,1)
			E_xmuLambda[i,j]=ximinusmj.T.dot(a[j]*inv(B[j])).dot(ximinusmj)+np.trace(a[j]*inv(B[j]).dot(sigma[j]))
	Eln_pLambda=np.multiply(a-d-1.0,Eln_Lambda)/2.0-0.5*np.asarray([np.trace(B[j].dot(a[j]*inv(B[j]))) for j in range(K)])
	Eln_pxc=np.sum(np.multiply(phi,0.5*Eln_Lambda+Eln_pi-0.5*E_xmuLambda),0)
	L=sum(Eln_ppi)+sum(Eln_pmu)+sum(Eln_pxc)+sum(Eln_pLambda)-Eln_qpi-sum(Eln_qmu)-sum(Eln_qLambda)
	return L

def gamma_d(a,d):
	j=np.arange(d)
	e=np.sum(gammaln((a+1.0)/2.0-j/2.0)-gammaln(a/2.0-j/2.0))
	g=np.exp(e)
	return g

def Gibbs():
	np.random.seed(1)
	dat=io.loadmat('data.mat')
	X=dat['X'].T
	N,d=X.shape
	T=500
	a_c=d
	c_c=0.1
	A=np.cov(X.T)
	B_c=c_c*d*A
	alpha_c=1.0
	m_c=np.mean(X,0)
	c_i=np.zeros(N)
	mu=np.zeros((N,d))
	Lamb=np.zeros((N,d,d))
	mu[0], Lamb[0]=parameters(0, X, c_i, c_c, m_c, a_c, B_c)
	prob_n=np.zeros((T,6))
	clusters=np.zeros(T)
	for t in range(T):
		phi=np.zeros((N,N))
		for i, xi in enumerate(X):
			n=np.asarray([len(np.where(c_i==z)[0]) if z!=c_i[i] else len(np.where(c_i==z)[0])-1 for z in range(N)])
			nJ=np.where(n>0)[0]
			for j in nJ:
				phi[i,j]=stats.multivariate_normal.pdf(xi, mean=mu[j], cov=inv(Lamb[j]))*n[j]/(alpha_c+N-1)

			jprime=max(set(c_i))+1
			c_ratio=c_c/(1.0+c_c)
			xminusm=(xi-m_c).reshape(d,1)
			marginal=(c_ratio/np.pi)**(d/2.0)*det(B_c+c_ratio*(xminusm).dot(xminusm.T))**(-0.5*(a_c+1.0))/det(B_c)**(-0.5*a_c)*gamma_d(a_c, d)
			phi[i,jprime]=alpha_c/(alpha_c+float(N)-1.0)*marginal			
			phi[i]=phi[i]/np.sum(phi[i])
			idx=np.where(phi[i]>0)[0]
			sample=stats.dirichlet.rvs(phi[i][phi[i]>0])
			cluster=idx[np.argmax(sample)]
			c_i[i]=cluster
			if c_i[i]==jprime:
				mu[jprime], Lamb[jprime]=parameters(jprime, X, c_i, c_c, m_c, a_c, B_c)
		for k in set(c_i):
			mu[k], Lamb[k]=parameters(k, X, c_i, c_c, m_c, a_c, B_c)
		n2=np.asarray(sorted([len(np.where(c_i==z)[0]) for z in range(N)]))[::-1]
		for q in range(6):
			if q<len(n2[n2>0]):
				prob_n[t,q]=n2[n2>0][q]
		clusters[t]=len(set(c_i))
	return phi, c_i, prob_n, clusters

def parameters(j, X, c_i, c_c, m_c, a_c, B_c):
	X_s=X[np.where(c_i==j)]
	s,d=X_s.shape
	m=float(c_c)/float(s+c_c)*m_c+1.0/float(s+c_c)*np.sum(X_s,0)
	c=s+c_c
	a=a_c+s
	xsum=0
	x_bar=np.mean(X_s,0)
	for i, xi in enumerate(X_s):
		xminusx_bar=(xi-x_bar).reshape(d,1)
		xsum=xsum+xminusx_bar.dot(xminusx_bar.T)
	x_barminusm=(x_bar-m_c).reshape(d,1)
	B=B_c+xsum+float(s)/float(a_c*s+1.0)*x_barminusm.dot(x_barminusm.T)
	Lamb=stats.wishart.rvs(a,inv(B))
	mu=stats.multivariate_normal.rvs(mean=m, cov=inv(c*Lamb))
	return mu, Lamb

def plot_scatter(X, K, phi, problem):
	np.random.seed(10)
	pred=np.argmax(phi,1)
	pred_list=np.unique(pred)
	fig=plt.figure()
	fig.suptitle('Problem %s.c K=%s' % (int(problem), K))
	col_list=colors.cnames.keys()
	col_list=[i for i in col_list if 'white' not in i and 'light' not in i]
	np.random.shuffle(col_list)
	markers=itertools.cycle(['x', 'd', '+', ',', '*', 'v', '^', '1', '.', 'o'])
	for k in pred_list:
		plt.scatter([j[0] for j in X[np.where(pred==k)]], [j[1] for j in X[np.where(pred==k)]], color=col_list[k], marker=markers.next())
	plt.legend(pred_list, fontsize=8, title='clusters', loc=4)	
	fig.savefig('Problem %sc K=%s.png' % (int(problem), K))		

def plot_n(prob_n):
	fig=plt.figure()
	fig.suptitle('Problem 3.b')
	plt.plot(prob_n)
	plt.xlabel('iteration t')
	plt.legend(['first', 'second', 'third', 'fourth', 'fifth', 'sixth'])
	fig.savefig('Problem 3b')

def plot_clusters(clusters):
	fig=plt.figure()
	fig.suptitle('Problem 3.c')
	plt.plot(np.arange(1,len(clusters)+1),clusters)
	plt.xlabel('iteration t')
	plt.ylim(0,max(clusters)+1)
	plt.legend(['clusters'])
	fig.savefig('Problem 3c')

def plot_L(L, K):
	fig=plt.figure()
	fig.suptitle('Problem 2.b K=%s' %K)
	plt.plot(L)
	plt.xlabel('iteration t')
	plt.ylabel('objctive function L')
	fig.savefig('Problem 2b K=%s' %K)

def plot_lnp(lnp, K):
	fig=plt.figure()
	fig.suptitle('Problem 1.b K=%s' %K)
	plt.plot(lnp)
	fig.savefig('Problem 1b K=%s.png' %K)	

def main():
	# PROBLEM 1
	K_list=[2,4,8,10]
	for k in K_list:
		print k
		phi,n,mu,Lamb,pi,lnp,X=EM_GMM(k)
		plot_lnp(lnp,k)
		plot_scatter(X,k,phi,1)
		
	# PROBLEM 2
	K_list=[2,4,10,25]
	for k in K_list:
		print k
		phi, alpha, sigma, m, n, a, B, L=VI(k)
		plot_L(L, k)
		plot_scatter(X,k,phi,2)

	# PROBLEM 3	
	phi,c_i,prob_n,clusters=Gibbs()
	plot_n(prob_n)
	plot_clusters(clusters)

if __name__ == '__main__':
	main()	


