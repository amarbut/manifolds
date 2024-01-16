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
#relative reconstruction error

def avg_reconstruction_error(data:list, quantizer):
    np_data = np.array(data)
    codes = quantizer.compute_codes(np_data)
    recon = quantizer.decode(codes)
    error = ((np_data - recon)**2).sum() / (np_data ** 2).sum()
    return error

#%%
#compare distribution (pairwise distances) of centroids

def compare_centroids(quantizer,m = "model_name", plot = False):
    
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
    
#%%
#compare dist of points assigned to each subspace

def compare_points(data, quantizer,m = "model_name", plot = False):
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
    all_hist = []
    for i in range(M):
        h = np.histogram(trans[i], bins = k)
        all_hist.append(h[0])
    # TODO: issue with plotting
    #     if plot = True:
    #         plt.bar(list(range(0,255)),np.sort(h[0])[::-1], alpha = 0.5)
    #         plt.title(m)
    # if plot = True:    
    #     plt.show()
        
    df = pd.DataFrame(all_hist).transpose()
        
    return df.describe()

# mean of histogram not meaningful...because same total points and bins. 

#%%
# compare dist of points within each cluster
# shows distribution of *average pairwise distances* between points in each
# cluster, for each subspace
# should show distribution of std dev instead?

def compare_point_dist(data, quantizer, m = "model_name", plot = False):
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

#%%
# nearest neighbor recall
# based on sablayrolles:
# https://github.com/facebookresearch/spreadingvectors/blob/main/lib/metrics.py


# def get_nearestneighbors_faiss(xq, xb, k, device, needs_exact=True, verbose=False):
#     assert device in ["cpu", "cuda"]

#     if verbose:
#         print("Computing nearest neighbors (Faiss)")

#     if needs_exact or device == 'cuda':
#         index = faiss.IndexFlatL2(xq.shape[1])
#     else:
#         index = faiss.index_factory(xq.shape[1], "HNSW32")
#         index.hnsw.efSearch = 64
#     if device == 'cuda':
#         index = faiss.index_cpu_to_all_gpus(index)

#     start = time.time()
#     index.add(xb)
#     _, I = index.search(xq, k)
#     if verbose:
#         print("  NN search (%s) done in %.2f s" % (
#             device, time.time() - start))

#     return I

# def evaluate(net, xq, xb, gt, quantizers, best_key, device=None,
#              trainset=None):
#     net.eval()
#     if device is None:
#         device = next(net.parameters()).device.type
#     xqt = forward_pass(net, sanitize(xq), device=device)
#     xbt = forward_pass(net, sanitize(xb), device=device)
#     if trainset is not None:
#         trainset = forward_pass(net, sanitize(trainset), device=device)
#     nq, d = xqt.shape
#     res = {}
#     score = 0
#     for quantizer in quantizers:
#         qt = getQuantizer(quantizer, d)

#         qt.train(trainset)
#         xbtq = qt(xbt)
#         if not qt.asymmetric:
#             xqtq = qt(xqt)
#             I = get_nearestneighbors(xqtq, xbtq, 100, device)
#         else:
#             I = get_nearestneighbors(xqt, xbtq, 100, device)

#         print("%s\t nbit=%3d: " % (quantizer, qt.bits), end=' ')

#         # compute 1-recall at ranks 1, 10, 100 (comparable with
#         # fig 5, left of the paper)
#         recalls = []
#         for rank in 1, 10, 100:
#             # number of points w/ 1,10,and 100 nn correct divided by total number points
#             recall = (I[:, :rank] == gt[:, :1]).sum() / float(nq)
#             key = '%s,rank=%d' % (quantizer, rank)
#             if key == best_key:
#                 score = recall
#             recalls.append(recall)
#             print('%.4f' % recall, end=" ")
#         res[quantizer] = recalls
#         print("")
        
#%%
# nearest neighbor recall
# my own implementation w/ FAISS

def NN_recall(data, quantizer, k, rank):
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
    