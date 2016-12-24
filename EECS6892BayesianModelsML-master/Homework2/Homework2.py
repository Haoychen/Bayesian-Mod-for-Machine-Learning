import scipy
from scipy import io, stats
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os


mat = scipy.io.loadmat('mnist_mat.mat')
Xtrain = mat['Xtrain']
ytrain = mat['ytrain']
Xtest = mat['Xtest']
ytest = mat['ytest']
Q = mat['Q']
d,n=Xtrain.shape
sigma=1.5
Lambda=1.0


def e_step(w):
	Xtrain1=Xtrain[:,np.where(ytrain==1)[1]]
	Xtrain0=Xtrain[:,np.where(ytrain==0)[1]]
	e=np.zeros(n)
	e[np.where(ytrain==1)[1]]=np.dot(Xtrain1.T,w)+sigma*np.divide(stats.norm.pdf(-np.dot(Xtrain1.T, w)/sigma),(1.0-stats.norm.cdf(-np.dot(Xtrain1.T,w)/sigma)))
	e[np.where(ytrain==0)[1]]=np.dot(Xtrain0.T,w)-sigma*np.divide(stats.norm.pdf(-np.dot(Xtrain0.T, w)/sigma),stats.norm.cdf(-np.dot(Xtrain0.T,w)/sigma))
	return e.reshape(n)

def m_step(E):
	xxT=np.sum(np.multiply(Xtrain.T.reshape(n,d,1),Xtrain.T.reshape(n,1,d)),0)
	xE=np.sum(np.multiply(Xtrain, E.reshape(1,n)), 1)
	w=np.dot(np.linalg.inv(Lambda*np.identity(d)+xxT/(sigma**2)),xE/(sigma**2)).reshape(d)
	return w

def W(T):
	w=np.tile(np.zeros((T+1,1)),d)
	E=np.tile(np.zeros((T+1,1)),n)
	for t in range(T):
		E[t+1]=e_step(w[t])
		w[t+1]=m_step(E[t+1])
	return w


def probit(w,t,X,y):
	e=d/2*math.log(Lambda/(2*math.pi))
	f=(Lambda/2)*np.dot(w[t].T, w[t])
	g=np.sum(np.multiply(1-y,np.log(1-stats.norm.cdf(np.dot(X.T,w[t])/sigma))))
	h=np.sum(np.multiply(y,np.log(stats.norm.cdf(np.dot(X.T,w[t])/sigma))))
	lnp=e-f+g+h
	return lnp
	
def plot_em(T):
	problem=r'2b: Plot lnp'
	w=W(T)
	lnp=[probit(w,t,Xtrain, ytrain) for t in range(T)]
	x=[t+1 for t in range(T)]	
	plt.plot(x, lnp)
	plt.title(problem)
	plt.xlabel(r't')
	plt.ylabel(r'ln[p(w_t, y | X)]')
	plt.interactive(True)	
	plt.savefig(problem.replace(':','').replace(' ', '_'))

def predictions(T):
	w=W(T)
	pr=stats.norm.cdf(np.dot(Xtest.T,w[T])/sigma)
	return pr

def guess(T):
	pr=predictions(T)
	k=len(pr)
	y_guess=np.zeros(k)
	y_guess[np.where(pr>=.5)]=1	
	y_guess[np.where(pr<.5)]=0
	return y_guess


def confusion_matrix(T):
	y_guess=guess(T)
	idx4=np.where(ytest[0]==0)
	idx9=np.where(ytest[0]==1)
	misclassified_4_idx=np.where(y_guess[idx4]==1)[0]
	classified_4_idx=np.where(y_guess[idx4]==0)[0]
	misclassified_9_idx=np.where(y_guess[idx9]==0)[0]
	classified_9_idx=np.where(y_guess[idx9]==1)[0]
	number_4_classified=classified_4_idx.size
	number_4_misclassified=misclassified_4_idx.size
	number_9_classified=classified_9_idx.size
	number_9_misclassified=misclassified_9_idx.size
	c_matrix=pd.DataFrame(columns=['Classified as 4', 'Classified as 9'], index=['ytest= 4', 'ytest= 9'])
	c_matrix['Classified as 4']['ytest= 4']=number_4_classified
	c_matrix['Classified as 9']['ytest= 4']=number_4_misclassified
	c_matrix['Classified as 9']['ytest= 9']=number_9_classified
	c_matrix['Classified as 4']['ytest= 9']=number_9_misclassified
	return c_matrix

def misclassified_digits(seed, T):
	probability=predictions(T)
	np.random.seed(seed)
	y_guess=guess(T)
	misclassified=ytest-y_guess
	m=np.where(misclassified!=0)[1]
	r=np.random.choice(m,3)
	problem='2d: Misclassified Predictions'
	image_prediction(r, y_guess, probability, problem)	

def ambiguous_predictions(T):
	probability=predictions(T)
	y_guess=guess(T)
	a=np.power(0.5-probability, 2)
	idx_min=a.argsort()[0:3]
	problem='2e: Ambiguous Predictions'
	image_prediction(idx_min, y_guess, probability, problem)	


def image_prediction(index_array, y_guess, probability, problem):	
	p={}
	fig = plt.figure(figsize=(15,7))
	fig.suptitle(problem, fontsize=18)
	for i in index_array:
		idx=list(index_array).index(i)
		binary_map=np.asarray([4,9])
		y_actual=ytest[0][i]	
		classification=binary_map[y_guess[i]]
		actual=binary_map[y_actual]
		img=np.reshape(np.dot(Q, Xtest)[:,i], (28,28))
		title='Pr(y*= %s |x*, Xtrain, ytrain) = %s \n Bayes Classifier Guess = %s \n ytest = %s' %(actual, '{:.4%}'.format(probability[i]), classification, actual)
		p[idx]=fig.add_subplot(1,3,idx+1)
		p[idx].set_title(title, fontsize=12)
		p[idx].axis('off')
		p[idx].imshow(img)
	plt.interactive(True)	
	fig.savefig(problem.replace(':','').replace(' ', '_'))


def w_image(t_list):	
	problem='2f: Reconstructing W'
	w=W(100)
	p={}
	fig = plt.figure(figsize=(15,7))
	fig.suptitle(problem, fontsize=18)
	for t in t_list:
		idx=list(t_list).index(t)
		img=np.reshape(np.dot(Q, w[t]), (28,28))
		title='Reconstructing W_%s' %t
		p[idx]=fig.add_subplot(2,3,idx+1)
		p[idx].set_title(title, fontsize=12)
		p[idx].axis('off')
		p[idx].imshow(img)
	plt.interactive(True)	
	fig.savefig(problem.replace(':','').replace(' ', '_'))	


def main(argv):
	i=int(argv[0])
	seed=int(argv[1])
	T=100
	if i in range(11,14):
		q='th' 
	elif i%10==1:
		q='st'
	elif i%10==2:
		q='nd'
	elif i%10==3:
		q='rd'
	else:
		q='th'	
	t_list=list((1,5,10,25,50,100))
	binary_map=np.asarray([4,9])
	y_guess=guess(T)
	k=len(y_guess)
	y_guess_map=np.zeros(k)
	y_guess_map[np.where(y_guess==0)]=4
	y_guess_map[np.where(y_guess==1)]=9
	plot_em(T)
	matrix=confusion_matrix(T)
	misclassified_digits(seed,T)
	ambiguous_predictions(T)
	w_image(t_list)
	sys.stdout=open('homework2.txt', 'w')
	print '2a: The %s-%s Xtest value is classified as %s.' % (i, q, binary_map[y_guess[i]])
	print '2b: Image has been saved as \'2b: Plot lnp\''
	os.system(r"2b_Plot_lnp.png")
	print '2c: The most probable label for each Xtest feature vector is \n    %s' % list(y_guess_map.astype(int))
	print '    The confusion matrix for correct classification/misclassification is as follows \n %s' %matrix
	print '2d: Image has been saved as \'2d: Misclassified Predictions.png\''
	os.system("2d_Misclassified_Predictions.png")
	print '2e: Image has been saved as \'2e: Ambiguous Predictions.png\''
	os.system("2e_Ambiguous_Predictions.png")
	print '2f: Image has been saved as \'2f: Reconstructing W.png\''
	os.system("2f_Reconstructing_W.png")
	sys.stdout.close()
	os.system("homework2.txt")


if __name__ == '__main__':
	main(sys.argv[1:])	
