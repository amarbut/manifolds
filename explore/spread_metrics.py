#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:33:36 2024

@author: anna
"""
from sklearn.decomposition import PCA
from sklearn.metrics import auc, pairwise
from scipy.stats import differential_entropy, multivariate_normal
import numpy as np
from math import e
import matplotlib.pyplot as plt  
from IsoScore.IsoScore import *

def normal_compare(dimension, size):
    #sample from a multivariate normal as a reference dist.
    x = multivariate_normal([0]*dimension, np.identity(dimension))
    y = x.rvs(size = size)
    pc = PCA()
    v = sum(pc.fit(y).explained_variance_)
    return x,y,v

def VRM(embeddings):
    N = len(embeddings) #number of embeddings
    d = len(embeddings[0]) #dimension of embeddings
    pc = PCA()
    tx = pc.fit_transform(embeddings) #PCA projection of embeddings
    
    #fix total explained variance in data to match that of a standard normal distribution
    comp,sample, ev = normal_compare(d,N)
    scale = ev/sum(pc.explained_variance_)
    tx *= np.sqrt(scale)
     
    #compute theoretical max vasicek entropy for std normal comparison
    m = np.log(np.sqrt(2*np.pi*e))
    
    ent = differential_entropy(tx)
    r = np.exp(ent)/np.exp(m)
    
    #report mean SSE from ratio of 1
    return round(np.sum((1-r)**2)/d,4)

def EEE(embeddings, plot = False):
    pc = PCA()
    pc.fit(embeddings)
    num_pc = pc.n_components_ #could also use dimension of embeddings
    
    #cumulative sum of exp. var of true components and uniform reference
    eigensum = np.cumsum(pc.explained_variance_) 
    ref = np.cumsum([eigensum[-1]/num_pc]*num_pc)
    
    #bring true cumulative sum to x axis (subtract diagonal) to find AUC
    AUC_sum = eigensum-ref 
    AUC = auc(range(num_pc),AUC_sum)
    
    total_poss = (eigensum[-1]*num_pc)/2 #total possible AUC
    
    if plot == True:
        plt.plot(range(num_pc), eigensum, label = 'Cumulative Sum of Eigenvalues')
        plt.plot(range(num_pc), ref)
        plt.legend()
        plt.show()
    
    #return AUC as a proportion of the total possible
    return round(AUC/total_poss,4) 

def mmd_rbf(embeddings,gamma=1.0):
    """ From https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    N = len(embeddings) #number of embeddings
    d = len(embeddings[0]) #dimension of embeddings
    comp,sample, ev = normal_compare(d,N) #draw from multivariate normal for ref.
    
    #compute gaussian kernel w/in and between observed and normal ref
    #K(x, y) = exp(-gamma ||x-y||^2) pairwise over x & y
    XX = pairwise.rbf_kernel(embeddings, embeddings, gamma)
    YY = pairwise.rbf_kernel(sample, sample, gamma)
    XY = pairwise.rbf_kernel(embeddings, sample, gamma)
    
    
    return XX.mean() + YY.mean() - 2 * XY.mean()

IsoScore(data)
