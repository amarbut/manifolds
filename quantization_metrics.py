#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 14:45:16 2023

@author: anna
"""

import faiss
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.special import kl_div
import time
#%%
#load sample spaces
model_samples = pickle.load(open("/media/anna/Samsung_T5/manifolds/ptb_model_samples_5Jun23.pkl", "rb"))

#%%
#build out quantized/clustered spaces
#kmeans
km = faiss.Kmeans(d = 768, k = 100)
km.train(model_samples['zhang'])

#product quantizer
pq = faiss.ProductQuantizer(768,4,8) #(dim, M subspaces, nbits)
pq.train( np.array(model_samples['zhang']))

#additive/local search quantizer
aq = faiss.LocalSearchQuantizer(768,4,8)
aq.train(np.array(model_samples['zhang']))


#%%


def avg_reconstruction_error(data:list, quantizer):
    #relative reconstruction error
    np_data = np.array(data)
    codes = quantizer.compute_codes(np_data)
    recon = quantizer.decode(codes)
    error = ((np_data - recon)**2).sum() / (np_data ** 2).sum()
    return error


def compare_centroids(quantizer,m = "model_name", plot = False):
    #compare distribution (pairwise distances) of centroids
    if hasattr(quantizer, 'centroids'):    
        c = faiss.vector_float_to_array(quantizer.centroids)
        M = quantizer.M
        k = quantizer.ksub
        d = quantizer.dsub
    elif hasattr(quantizer, 'codebooks'):
        c = faiss.vector_float_to_array(quantizer.codebooks)
        M = quantizer.M
        k = quantizer.K
        d = quantizer.d
    else:
        raise Exception("Can't find centroids")

    c = c.reshape(M,k,d)
    
    all_dist = []
    for i in range(M):
        c_dist = pairwise_distances(c[i]).flatten()
        all_dist.append(c_dist)
    # TODO: issues with plotting
    # AttributeError: module 'PIL' has no attribute 'Image'
    #     if plot == True:
    #         sns.kdeplot(c_dist)
    #         plt.title(m)
    # if plot == True:    
    #     plt.show()
        
    df = pd.DataFrame(all_dist).transpose()
        
    return df.describe()


def compare_points(data, quantizer,m = "model_name", plot = False):
    #compare dist of points assigned to each subspace
    if hasattr(quantizer, 'centroids'):    
        M = quantizer.M
        k = quantizer.ksub
    elif hasattr(quantizer, 'codebooks'):
        M = quantizer.M
        k = quantizer.K
    else:
        raise Exception("Can't find quantizer attributes")
    
    codes = quantizer.compute_codes(data)
    trans = codes.T
    all_hist = []
    for i in range(M):
        h = np.histogram(trans[i], bins = k)
        all_hist.extend(list(h[0]))
    # TODO: issue with plotting
    #     if plot = True:
    #         plt.bar(list(range(0,255)),np.sort(h[0])[::-1], alpha = 0.5)
    #         plt.title(m)
    # if plot = True:    
    #     plt.show()
        
    var = np.var(all_hist)
    
    #sample from uniform to compare in kl_div
    u = np.random.uniform(0, k, len(data)*M) #duplicate data to replicate subspaces
    u_hist = np.histogram(u, bins = k*M)[0]
    kl = sum(kl_div(all_hist, u_hist)) #elementwise function needs to be summed
    
    
    return var,kl


def compare_point_dist(data, quantizer, m = "model_name", plot = False):
    # compare dist of points within each cluster
    # shows distribution of *average pairwise distances* between points in each
    # cluster, for each subspace
    # should show distribution of std dev instead?
    if hasattr(quantizer, 'centroids'):    
        M = quantizer.M
        k = quantizer.ksub
        d = quantizer.dsub
    elif hasattr(quantizer, 'codebooks'):
        M = quantizer.M
        k = quantizer.K
        d = quantizer.d
    else:
        raise Exception("Can't find quantizer attributes")
    
    codes = quantizer.compute_codes(data)
    trans = codes.T
    
    all_dist = []
    for i in range(M):
        #make list of original vectors per centroid
        c_list = [[] for i in range(k)]
        d = []
        for idx, v in enumerate(data):
            c = trans[i][idx]
            c_list[c-1].append(v)
            
        # get average pairwise distance between points assigned to each centroid
        for j in c_list:
            c_dist = pairwise_distances(j).flatten()
            avg = np.mean(c_dist)
            d.append(avg)
            
        all_dist.append(d)
    
    # TODO: issues with plotting 
    #     if plot == True:
    #         #visualize avg pairwise distance for points in each centroid for each subspace
    #         sns.kdeplot(d)
    #         plt.title(m)
    # if plot == True:    
    #     plt.show()
    
    df = pd.DataFrame(all_dist).transpose()
        
    return df.describe()


def NN_recall(data, quantizer, k, rank):
    # nearest neighbor recall
    # my own implementation w/ FAISS
    m = len(data[0]) #vector length
    N = len(data) #number points
    
    #compute nn for original data
    index_t = faiss.IndexFlatL2(m)
    index_t.add(data)
    D_t,I_t = index_t.search(data, k+1)
    
    #compute nn for reconstructed data
    codes = quantizer.compute_codes(data)
    q_data = quantizer.decode(codes)
    
    index_c = faiss.IndexFlatL2(m)
    index_c.add(q_data)
    D_c,I_c = index_c.search(q_data, k+1)
    
    #count number of points with exact matching knn (non-ordered)
    #match_count = sum([set(I_c[i,:rank])==set(I_t[i,:rank]) for i in range(N)])
    
    #count number of points with first nn in top k (traditional recall@k)
    match_count = (I_c[:,:rank] == I_t[:,1:2]).sum() #FAISS puts original point as NN slot 0
    
    recall = match_count/float(N)
    
    return recall
    

def patchiness(data, quantizer):
    #patchiness index to be applied to quantized datapoints
    n = len(data)#/M?? NO, quantizer sees all points, but assigns them over M subspaces which are later recombined
    if hasattr(quantizer, 'centroids'):
        k = quantizer.ksub
    elif hasattr(quantizer, 'codebooks'):
        k = quantizer.K
    else:
        raise Exception("Can't find quantizer attributes")
    
    m = n/k #avg points per cell
    V, kl = compare_points(data, quantizer) #variance in m over all cells
    
    m_star = m +((V/m)-1) #density from individual's perspective
    
    return m_star/m #patchiness index