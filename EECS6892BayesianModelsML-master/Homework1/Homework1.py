import scipy
from scipy import io, stats
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os

mat = scipy.io.loadmat('mnist_mat.mat')

def prior(y):
	e=1.0
	f=1.0
	prior={}
	N=mat['ytest'].size
	if y==1:
		prior=float(e+np.where(mat['ytrain']==y)[0].size)/float(N+e+f)
	if y==0:
		prior=float(f+np.where(mat['ytrain']==y)[0].size)/float(N+e+f)
	return prior

def likelihood(y, i):
	a=1.0
	b=1.0
	c=1.0
	p=mat['Xtrain'].shape[0]
	stdev=np.std(np.hstack((mat['Xtrain'],mat['Xtest'])),1).reshape(p, 1)
	Xtest=mat['Xtest']/stdev
	#Xtest=mat['Xtest']
	x_new=Xtest[:,i]
	Xtrain=mat['Xtrain']/stdev
	#Xtrain=mat['Xtrain']
	x=Xtrain[:,np.where(mat['ytrain']==y)[1]]
	n=x.shape[1]
	v=2.0*b+n
	scale=2*float((a*n+a+1))/float((v+a*n*v))
	mean=np.mean(x,1)
	mu=n*mean/float((1/float(a)+n))
	t=sum(-(v+1)/2*np.log(1+np.power(np.subtract(x_new,mu),2)/(scale*v)))
	return t

def posterior(y, i):
	l=likelihood(y, i)
	p=prior(y)
	pr_y=l*p
	return pr_y

def predictions():
	r=mat['ytest'].size
	y_guess=np.zeros(r)
	for i in range(r):
		pr_0=posterior(0, i)
		pr_1=posterior(1, i)
		y_hat=max(pr_0, pr_1)
		if y_hat==pr_0:
			y_guess[i]=0
		if y_hat==pr_1:
			y_guess[i]=1
	return y_guess

def confusion_matrix():
	y_guess=predictions()
	idx4=np.where(mat['ytest'][0]==0)
	idx9=np.where(mat['ytest'][0]==1)
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

def misclassified_digits(seed):
	probability=prediction_probability()
	np.random.seed(seed)
	y_guess=guess()
	misclassified=mat['ytest'][0]-y_guess
	m=np.where(misclassified!=0)[0]
	r=np.random.choice(m,3)
	problem='4c: Misclassified Predictions'
	image_prediction(r, y_guess, probability, problem)

def ambiguous_predictions():
	probability=prediction_probability()
	y_guess=predictions()
	a=np.power(0.5-probability[0], 2)
	idx_min=a.argsort()[0:3]
	problem='4d: Ambiguous Predictions'
	image_prediction(idx_min, y_guess, probability, problem)

def prediction_probability():
	r=mat['ytest'].size
	pr_0=np.zeros(r)
	pr_1=np.zeros(r)
	for i in range(r):
		pr_0[i]=posterior(0, i)
		pr_1[i]=posterior(1, i)
	pr=np.vstack((pr_0, pr_1))
	probability=1-pr/np.sum(pr, 0)
	return probability

def image_prediction(index_array, y_guess, probability, problem):	
	p={}
	fig = plt.figure(figsize=(15,7))
	fig.suptitle(problem, fontsize=18)
	for i in index_array:
		idx=list(index_array).index(i)
		binary_map=np.asarray([4,9])
		y_actual=mat['ytest'][0][i]	
		classification=binary_map[y_guess[i]]
		actual=binary_map[y_actual]
		img=np.reshape(np.dot(mat['Q'], mat['Xtest'])[:,i], (28,28))
		title='Pr(y*= %s |x*, Xtrain, ytrain) = %s \n Bayes Classifier Guess = %s \n ytest = %s' %(actual, '{:.4%}'.format(probability[y_actual][i]), classification, actual)
		p[idx]=fig.add_subplot(1,3,idx+1)
		p[idx].set_title(title, fontsize=12)
		p[idx].axis('off')
		p[idx].imshow(img)
	plt.interactive(True)	
	fig.savefig(problem.replace(':','').replace(' ', '_'))

	
def main(argv):
	i=int(argv[0])
	seed=int(argv[1])
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
	binary_map=np.asarray([4,9])
	y_guess=predictions()
	n=len(y_guess)
	y_guess_map=np.zeros(n)
	y_guess_map[np.where(y_guess==0)]=4
	y_guess_map[np.where(y_guess==1)]=9
	matrix=confusion_matrix()
	misclassified_digits(seed)
	ambiguous_predictions()
	sys.stdout=open('homework1.txt', 'w')
	print '4a: The %s-%s Xtest value is classified as %s.' % (i, q, binary_map[y_guess[i]])
	print '4b: The most probable label for each Xtest feature vector is \n    %s' % y_guess_map.astype(int)
	print '    The confusion matrix for correct classification/misclassification is as follows \n %s' %matrix
	print '4c: Image has been saved as \'4c: Misclassified Predictions.png\''
	os.system("4c_Misclassified_Predictions.png")
	print '4d: Image has been saved as \'4d: Ambiguous Predictions.png\''
	os.system("4d_Ambiguous_Predictions.png")
	sys.stdout.close()
	os.system("homework1.txt")


if __name__ == '__main__':
	main(sys.argv[1:])	
