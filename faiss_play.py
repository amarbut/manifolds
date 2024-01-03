#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:20:00 2023

@author: anna
"""

import faiss
import numpy as np
import pickle

?faiss.Kmeans

model_samples = pickle.load(open("ptb_model_samples_5Jun23.pkl", "rb"))

km = faiss.Kmeans(d = 768, k = 100)
km.train(model_samples['zhang'])

?faiss.ProductQuantizer

pq = faiss.ProductQuantizer(768,4,8) #(dim, M subspaces, nbits)
pq.train( np.array(model_samples['zhang']))

#%%
#compare error when using quantizer trained on base BERT on other Zhang models

codes = pq.compute_codes(np.array(model_samples['zhang']))
x2 = pq.decode(codes)
avg_relative_error = ((np.array(model_samples['zhang']) - x2)**2).sum() / (np.array(model_samples['zhang']) ** 2).sum()

codes2 = pq.compute_codes(np.array(model_samples['germ']))
x3 = pq.decode(codes2)
avg_relative_error2 = ((np.array(model_samples['germ']) - x3)**2).sum() / (np.array(model_samples['germ']) ** 2).sum()

codes3 = pq.compute_codes(np.array(model_samples['chin']))
x4 = pq.decode(codes3)
avg_relative_error3 = ((np.array(model_samples['chin']) - x4)**2).sum() / (np.array(model_samples['chin']) ** 2).sum()

codes4 = pq.compute_codes(np.array(model_samples['untrained']))
x5 = pq.decode(codes4)
avg_relative_error4 = ((np.array(model_samples['untrained']) - x5)**2).sum() / (np.array(model_samples['untrained']) ** 2).sum()

#random model equal with other ling models
#%%
# compare error on quantizers trained separately for ea model

for m in ['zhang', 'germ', 'chin', 'untrained']:
    x = np.array(model_samples[m])
    pq = faiss.ProductQuantizer(768,4,8)
    pq.train(x)
    
    codes = pq.compute_codes(x)
    x2 = pq.decode(codes)
    print(m)
    print(((x - x2)**2).sum() / (x ** 2).sum())
    
# zhang
# 0.13883212
# germ
# 0.075298704
# chin
# 0.0705675
# untrained
# 0.3396936

#%%
#find quantizer centroids
c = faiss.vector_float_to_array(pq.centroids)
M = pq.M
k = pq.ksub
d = pq.dsub
c = c.reshape(M,k,d)
#%%
#compare distribution of centroids in each subspace for each model
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

for m in ['zhang', 'germ', 'chin', 'untrained']:
    x = np.array(model_samples[m])
    pq = faiss.ProductQuantizer(768,4,8)
    pq.train(x)
    
    c = faiss.vector_float_to_array(pq.centroids)
    M = pq.M
    k = pq.ksub
    d = pq.dsub
    c = c.reshape(M,k,d)
    
    for i in range(M):
        c_dist = pairwise_distances(c[i]).flatten()
        sns.kdeplot(c_dist)
        plt.title(m)
        
    plt.show()
#%%
# compare dist of points assigned to each centroids per subspace for each model

for m in ['zhang', 'germ', 'chin', 'untrained']:
    x = np.array(model_samples[m])
    pq = faiss.ProductQuantizer(768,4,8)
    pq.train(x)
    M = pq.M
    
    codes = pq.compute_codes(x)
    trans = codes.T
    for i in range(M):
        h = np.histogram(trans[i], bins = 255)
        plt.bar(list(range(0,255)),np.sort(h[0])[::-1], alpha = 0.5)
        plt.title(m)
        
    plt.show()
    
#%%
#compare pairwise distances between points assigned to each centroid per subspace per model

for m in ['zhang', 'germ', 'chin', 'untrained']:
    x = np.array(model_samples[m])
    pq = faiss.ProductQuantizer(768,4,8)
    pq.train(x)
    M = pq.M
    
    codes = pq.compute_codes(x)
    trans = codes.T
    
    for i in range(M):
        #make list of original vectors per centroid
        c_list = [[] for i in range(256)]
        d = []
        for idx, v in enumerate(x):
            c = trans[i][idx]
            c_list[c-1].append(v)
            
        # get average pairwise distance between points assigned to each centroid
        for j in c_list:
            c_dist = pairwise_distances(j).flatten()
            avg = np.mean(c_dist)
            d.append(avg)
            
        #visualize avg pairwise distance for points in each centroid for each subspace
        sns.kdeplot(d)
        plt.title(m)
        
    plt.show()
            

