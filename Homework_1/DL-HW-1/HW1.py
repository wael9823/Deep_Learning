#!/usr/bin/env python3

# First Problem of HW1 of Deep Learning
# Author = "Wael Mohammed"

import numpy as np

def problem_1a(A,B):
    return(A+B)

def problem_1b(A,B,C):
    return(np.dot(A,B)-C)

def problem_1c(A,B,C):
    return(A*B+C.T)

def problem_1d(x,y):
    return(np.dot(x.T,y))

def problem_1e(A,x):
    return(np.linalg.solve(A,x))

def problem_1f(A,x):
    return(np.linalg.solve(A.transpose(),x.transpose()).transpose())

def problem_1g(A,i):
    return(np.sum(A[i,::2]))

def problem_1h(A,c,d):
	return(np.mean(A[np.nonzero((A >= c) & (A<= d))]))

def problem_1i(A,k):
    eigenvalues,eigenvectors = np.linalg.eig(A)
    x = np.argsort(eigenvalues)[-k:]
    return (eigenvectors[x,:])

def problem_1j(x,k,m,s):
    z = np.ones((x.shape[0],1))
    mean = (x+m*z).flatten()
    cov = s*np.identity(x.shape[0])
    return (np.random.multivariate_normal(mean, cov,k).T)

def problem_1k(A):
    np.random.shuffle(A)
    return (A)

def problem_1l(x):
    return ([(value - np.mean(x)) / np.std(x) for value in x])

def problem_1m(x,k):
    return (np.repeat(x,k,1))

def problem_1n(A):
    b = np.repeat(A[:, :, np.newaxis], A.shape[1], axis=2)
    c = np.transpose(b,(0,2,1))
    d = (b-c)**2
    return (np.sqrt(np.sum(d,0))) #Implement function


def linear_regression (X_tr, y_tr):
    term1 = np.linalg.inv(np.dot(X_tr.T,X_tr))
    term2 = np.dot(X_tr.T,y_tr)
    return(np.dot(term1,term2))
 

def train_age_regressor ():
    # Load data
    
    X_tr = np.reshape(np.load("Homework 1/data/age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("Homework 1/data/age_regression_ytr.npy")
    X_te = np.reshape(np.load("Homework 1/data/age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("Homework 1/data/age_regression_yte.npy")
    print(X_tr.shape)
    print(ytr.shape)
    w = linear_regression(X_tr, ytr)
    # Report fMSE cost on the training and testing data (separately)
    print(w)
    print(w.shape)
    # print(np.sum(np.dot(X_tr,w)-ytr))

    print((np.sum((np.dot(X_tr,w)-ytr)**2))/(2*X_tr.shape[0]))
    print((np.sum((np.dot(X_te,w)-yte)**2))/(2*X_te.shape[0]))

train_age_regressor()
