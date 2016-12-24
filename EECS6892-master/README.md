# EECS6892: Bayesian Methods for Machine Learning
-----

This repository will contain the homework solutions for my Bayesian ML class taught by John Paisley at Columbia University in the Fall of 2015. The solutions here are written in Python.

[Course homepage](http://www.columbia.edu/~jwp2128/Teaching/E6892/E6892Fall2015.html)

## Homework 1

The first three problems are mostly mathematical, the last question is data-centric and is a MAP method on the MNIST dataset.

    1. Gameshow probability puzzle
    2. Posterior of Multinomial with a conjugate prior
    3. Deriving a posterior for a Normal with a Normal prior on the mean and Gamma prior on the variance
    4. Naive Bayes classifier using derivation of 3.

## Homework 2

The first problem was a derivation of EM for a (d x k) Gasussian Matrix Posterior distribution. The second problem was an implemenation of EM for Probit regression. 

    1. EM for a Matrix (pure derivation)
    2. This is an implementation of Probit Regression using EM

## Homework 3

The first problem was a derivation of Variational Inference for a regresion model with a Sparse ARD prior.

    1. A regression model with two gamma priors on the variance and on the diagonal of the variance of the weight matrix
    2. This is implementation of the derived model in 1 with various settings of the model which yields a sparse model

## Homework 4

The problem was on implementing three different algorithms for Guassian Mixture models.

    1. EM to learn the GMM
    2. Varitaional Inference to learn the GMM
    3. Bayesian non-parametric sampler for a marginalized GMM (using an infinite Dirichlet Prior)

