#Allison Fenichel 
#Homework3
#EECS6892


import scipy
from scipy import io, stats
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os




def vars(n):
	exec('dat=scipy.io.loadmat(\'data%s.mat\')' % n)
	exec('X=dat[\'X\'].T')
	exec('y=dat[\'y\']')
	exec('z=dat[\'z\']')
	return X, y, z

def iterations(X, y, T):
	d,n=X.shape
	e0=1.0
	f0=1.0
	a0=np.full(d, 10e-16)
	b0=np.full(d, 10e-16)
	mu0=np.ones((d,1))
	sigma0=np.ones((d,d))
	L=np.zeros(T)
	Eln_qw=np.zeros(T)
	Eln_qlambda=np.zeros(T)
	Eln_qalpha=np.zeros(T)
	Eln_pw=np.zeros(T)
	Eln_plambda=np.zeros(T)
	Eln_palpha=np.zeros(T)
	Eln_py=np.zeros(T)
	for t in range(T):
		a,b,e,f,mu,sigma=updates(X,y,a0,b0,e0,f0,mu0,sigma0)
		L[t]=obj_function(X,y,a,b,e,f,mu,sigma) 
		if t<T:
			a0=a
			b0=b
			e0=e
			f0=f
			mu0=mu
			sigma0=sigma
	return a, b, e, f, L, mu


def updates(X,y,a0,b0,e0,f0,mu0,sigma0):
	d,n=X.shape
	xTsigmax=0
	xxT=0
	yminusxTmu=0
	for i in range(n):
		xTsigmax=xTsigmax+np.dot(np.dot(X[:,i].reshape(1,d), sigma0),X[:,i].reshape(d,1))
		xxT=xxT+np.dot(X[:,i].reshape(d,1), X[:,i].reshape(1,d))
		yminusxTmu=yminusxTmu+np.power(y[i]-X[:,i].T.dot(mu0),2)
	a=a0+0.5
	b=np.zeros(d)
	for k in range(d):
		b[k]=b0[k]+0.5*(sigma0[k][k]+mu0[k]**2)
	e=e0+float(n)/2.0
	f=f0+0.5*(yminusxTmu+xTsigmax)
	sigma=np.linalg.inv((e/f[0][0])*xxT+np.diag(a/b))
	mu=np.dot(sigma, e/f[0][0]*np.sum(np.multiply(y.reshape(1,n),X),1))
	return a,b,e,f,mu,sigma

def obj_function(X,y,a,b,e,f,mu,sigma): 
	d,n=X.shape
	e0=1.0
	f0=1.0
	a0=np.full(d, 10e-16)
	b0=np.full(d, 10e-16)
	
	Eln_qw=0.5*np.log(np.linalg.det(sigma))
	Eln_qlambda=e-np.log(f)+scipy.special.gammaln(e)+np.multiply(1.0-e,scipy.special.psi(e))
	Eln_qalpha=sum(a-np.log(b)+scipy.special.gammaln(a)+np.multiply(1.0-a,scipy.special.psi(a)))
	diag_ab=np.diag(a/b)
	Eln_pw=0.5*sum(scipy.special.psi(a)-np.log(b))-0.5*np.trace(np.dot(diag_ab, sigma))-0.5*np.dot(np.dot(mu.T, diag_ab),mu) 
	Eln_plambda=(e0-1.0)*(scipy.special.psi(e)-np.log(f))-np.multiply(f0,e/f[0][0])
	Eln_palpha=sum((a0-1.0)*(scipy.special.psi(a)-np.log(b))-np.multiply(b0,np.divide(a,b)))
	xTsigmax=0
	yminusxTmu=0
	for i in range(n):
		xTsigmax=xTsigmax+np.dot(np.dot(X[:,i].reshape(1,d), sigma),X[:,i].reshape(d,1))
		yminusxTmu=yminusxTmu+np.power(y[i]-X[:,i].T.dot(mu),2)
	Eln_py=float(n)/2*(scipy.special.psi(e)-np.log(f))-0.5*e/f*(yminusxTmu+xTsigmax)
	L=Eln_plambda+Eln_pw+Eln_palpha+Eln_py-Eln_qlambda-Eln_qw-Eln_qalpha
	return L
	
def plotf(v, p):
	fig1=plt.figure()
	fig1.suptitle('Problem 2a: Data Set %s' % p)
	plt.plot(v)
	plt.xlabel('Iteration t')
	plt.ylabel('L')
	fig1.savefig('Problem2a_data%s' % p)

def stemplotf(v, p):
	fig2=plt.figure()
	fig2.suptitle('Problem 2b: Data Set %s' % p)
	plt.stem(v)
	plt.xlabel('k')
	plt.ylabel('1/E[alpha_k]')
	fig2.savefig('Problem2b_data%s' % p)



def distributionplot(X,mu,y,z,p):
	yhat = X.T.dot(mu)
	z_n = np.linspace(-5,5,len(y))
	fz_n = 10.0*np.sinc(z_n)
	fig3=plt.figure()
	fig3.suptitle('Problem 2d: Data Set %s' % p)
	plt.plot(z,yhat,color='orange',label='Predicted values y_hat')
	plt.scatter(z,y,color='grey',label='Actual values y')
	plt.plot(z_n,fz_n,color='blue',label='True values 10*sinc(z)')
	plt.legend()
	fig3.savefig('Problem2d_data%s' % p)


def main():
	k=3
	e_p={}
	f_p={}
	for i in range(k):
		p=i+1
		X, y, z=vars(p)
		a, b, e, f, L, mu=iterations(X, y, 500)
		plotf(L, p)
		stemplotf(b/a, p)
		distributionplot(X,mu,y,z,p)
		e_p[p]=e
		f_p[p]=f[0][0]
		
	f=open('homework3.txt', 'w')
	f.writelines('2a: \nImages of the objective function have been saved as \'Problem2a_data1.png\', \'Problem2a_data2.png\', and \'Problem2a_data3.png\' for each respective data set\n\n')
	f.writelines('2b: \nImages of 1/E[alpha_k] as a function of k have been saved as \'Problem2b_data1.png\', \'Problem2b_data2.png\', and \'Problem2b_data3.png\' for each respective data set\n\n')
	f.writelines('2c:\n')
	for i in range(k):
	 	p=i+1
	 	f.write('For data set %s, the final value for 1/E[lambda] is %s\n' % (p, f_p[p]/e_p[p]))
	f.writelines('\n2d: \nPlot images have been saved as \'Problem2d_data1.png\', \'Problem2d_data2.png\', and \'Problem2d_data3.png\' for each respective data set\n\n')
	f.close()
	

if __name__ == '__main__':
	main()	



		