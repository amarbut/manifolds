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
model_samples = pickle.load(open("/media/anna/Samsung_T5/manifolds/zhang1_samplespace_14Mar24.pkl", "rb"))

#%%
#build out quantized/clustered spaces
#kmeans
# km = faiss.Kmeans(d = 768, k = 100)
# km.train(model_samples['zhang'])

#product quantizer
pq = faiss.ProductQuantizer(768,4,8) #(dim, M subspaces, nbits=256 centroids)
pq.train( np.array(model_samples))

#additive/local search quantizer
aq = faiss.LocalSearchQuantizer(768,4,8)
aq.train(np.array(model_samples))


#%%

def avg_reconstruction_error(data:list, quantizer):
    #relative reconstruction error
    np_data = np.array(data)
    codes = quantizer.compute_codes(np_data)
    recon = quantizer.decode(codes)
    error = ((np_data - recon)**2).sum() / (np_data ** 2).sum()
    return error


def compare_centroids(data, quantizer,m = "model_name", plot = False):
    #compare distribution (pairwise distances) of centroids
    #report var of normalized NN distance for each centroid
    #also report normalized var of distance to farthest point for ea. centroid
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
    
    #compute NN distance for all centroids, then normalize
    nn = []
    for i in range(M):
        index = faiss.IndexFlatL2(d)
        index.add(np.array(c[i]).astype('float32'))
        D,I = index.search(np.array(c[i]).astype('float32'), 2)
        nn_dist = [D[j,1] for j in range(k)]
        nn_dist /= np.max(nn_dist)
        nn.extend(nn_dist)
        
    
    #compute pairwise distance between all centroids
    # all_dist = []
    # for i in range(M):
    #     c_dist = pairwise_distances(c[i]).flatten()
    #     all_dist.append(c_dist)
    # df = pd.DataFrame(all_dist).transpose()
    
    codes = quantizer.compute_codes(data)
    trans = codes.T
    
    
    for i in range(M):
        #make list of original vectors per centroid
        c_list = [[] for j in range(k)] #initialize empty list for each centroid
        for idx, v in enumerate(data):
            j = trans[i][idx] #for each vector, find assigned centroid index
            c_list[j].append(np.linalg.norm(c[i][j]-v)) #add l2 dist from centroid to point
    
    max_dist = [np.max(i) for i in c_list] #report max dist to assigned point for ea centroid
    max_dist = [i/np.max(max_dist) for i in max_dist] #normalize max distances
    
    #returns list of nn distances, var of nn, list of max pt distances, var of max dist
    return nn, np.var(nn), max_dist, np.var(max_dist)


def compare_points(data, quantizer,m = "model_name", plot = False):
    #compare dist of number points assigned to each subspace
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
    
    #sample from uniform to compare in kl_div
    u = np.random.uniform(0, k, len(data)) #duplicate data to replicate subspaces
    u_hist = np.histogram(u, bins = k)[0]
    u_hist = [i/(len(data)) for i in u_hist]
    
    metrics = []
    all_hist = []
    for i in range(M):
        h = np.histogram(trans[i], bins = k)[0]
        h = [i/(len(data)) for i in h]
        var = np.var(h)
        kl = sum(kl_div(h, u_hist)) #elementwise function needs to be summed
        metrics.append((var,kl))
        all_hist.extend(h)
    # TODO: issue with plotting
    #     if plot = True:
    #         plt.bar(list(range(0,255)),np.sort(h[0])[::-1], alpha = 0.5)
    #         plt.title(m)
    # if plot = True:    
    #     plt.show()
    
    #report average var and kl over subspaces
    avg_var = np.mean([k[0] for k in metrics])
    avg_kl = np.mean([k[1] for k in metrics])        
    
    #returns single list of centroid-datapoint counts, avg var and kl over subspaces
    return all_hist, avg_var, avg_kl


def compare_point_dist(data, quantizer, m = "model_name", plot = False):
    # compare dist of points within each cluster
    # report EEE for each centroid/cluster and average EEE for all
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
    
    all_EEE = []
    all_c = []
    for i in range(M):
        #make list of original vectors per centroid
        c_list = [[] for j in range(k)]
        for idx, v in enumerate(data):
            c = trans[i][idx]
            c_list[c-1].append(v)
            
        # get average pairwise distance between points assigned to each centroid
        # for j in c_list:
        #     c_dist = pairwise_distances(j).flatten()
        #     avg = np.mean(c_dist)
        #     d.append(avg)
            
        # all_dist.append(d)
        
        #get EEE for data in cluster
        c_list = [np.unique(k, axis = 0) for k in c_list] #control for identical embeddings
        e = [EEE(k) for k in c_list if len(k) > 1]
        all_EEE.extend(e)
        all_c.append(c_list)
        
    
    # TODO: issues with plotting 
    #     if plot == True:
    #         #visualize avg pairwise distance for points in each centroid for each subspace
    #         sns.kdeplot(d)
    #         plt.title(m)
    # if plot == True:    
    #     plt.show()
    
    #df = pd.DataFrame(all_dist).transpose()
    # returns list of EEE scores for all clusters, avg EEE of all clusters
    return all_EEE, np.mean(all_EEE)

def NN_recall(data, quantizer, k, rank):
    # nearest neighbor recall
    # my own implementation w/ FAISS
    m = len(data[0]) #vector length
    N = len(data) #number points
    
    #compute nn for original data
    print("nn for original")
    index_t = faiss.IndexFlatL2(m)
    index_t.add(data)
    D_t,I_t = index_t.search(data, k+1)
    
    print("reconstruct data")
    #compute nn for reconstructed data
    codes = quantizer.compute_codes(data)
    q_data = quantizer.decode(codes)
    
    print("nn for reconstruct")
    index_c = faiss.IndexFlatL2(m)
    index_c.add(q_data)
    D_c,I_c = index_c.search(q_data, k+1)
    
    #count number of points with exact matching knn (non-ordered)
    #match_count = sum([set(I_c[i,:rank])==set(I_t[i,:rank]) for i in range(N)])
    
    #count number of points with first nn in top k (traditional recall@k)
    match_count = (I_c[:,:rank] == I_t[:,1:2]).sum() #FAISS puts original point as NN slot 0
    
    recall = match_count/float(N)
    
    return recall
    

def patchiness(data, quantizer, c, w):
    # c = counts of datapoints assigned to ea cell
    # w = list of max dist to assigned datapoint per cell
    #patchiness index to be applied to quantized datapoints
    n = len(data)#/M?? NO, quantizer sees all points, but assigns them over M subspaces which are later recombined
    if hasattr(quantizer, 'centroids'):
        k = quantizer.ksub
    elif hasattr(quantizer, 'codebooks'):
        k = quantizer.K
    else:
        raise Exception("Can't find quantizer attributes")
    
    m = sum(w)/k #weighted avg proportion of points per cell
    V = np.var([(c[i]*w[i])/n for i in range(k)]) #variance of weighted cell densities
    m_star = m +((V/m)-1) #density from individual's perspective
    
    return m_star/m #patchiness index