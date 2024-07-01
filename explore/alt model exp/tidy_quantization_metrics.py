#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:08:07 2024

@author: anna
"""
import faiss
import numpy as np
from scipy.special import kl_div


class quantization:
    def __init__(self, data, quantizer):
        self.quantizer = quantizer
        self.data = data
        self.n = len(data)
        
        if hasattr(quantizer, 'centroids'):    
            c = faiss.vector_float_to_array(quantizer.centroids)
            self.M = quantizer.M
            self.k = quantizer.ksub
            self.d = quantizer.dsub
        elif hasattr(quantizer, 'codebooks'):
            c = faiss.vector_float_to_array(quantizer.codebooks)
            self.M = quantizer.M
            self.k = quantizer.K
            self.d = quantizer.d
        else:
            raise Exception("Can't find centroids")
        
        #"codebook" or list of cluster centroids
        self.c = c.reshape(self.M,self.k,self.d)
        
        #assign points from dataset to centroids
        self.codes = quantizer.compute_codes(data)
        self.trans = self.codes.T
        
        #make list of original vectors per centroid
        self.all_c = []
        for i in range(self.M):
            c_list = [[] for j in range(self.k)]
            for idx, v in enumerate(data):
                c = self.trans[i][idx] # find assigned centroid index for ea. token
                c_list[c-1].append(v) # add token vector to centroid list
            
            c_list = [np.unique(j, axis = 0) for j in c_list] #control for identical embeddings
            self.all_c.append(c_list)
        
        
        
    def avg_reconstruction_error(self):
        # average reconstruction error between true and reconstructed data
        recon = self.quantizer.decode(self.codes)
        error = ((data - recon)**2).sum() / (data ** 2).sum()
        return error
    
    
    
    def compare_centroids(self):
        # considering distribution of centroids
        # returns list of normalized distances to nearest centroid per centroid
        # returns list of max distance to assigned data pont per centroid
        nn = []
        #max_dist = []
        for i in range(self.M):
            # use FAISS L2 nn index to find nearest centroid
            array_c = np.array(self.c[i]).astype('float32')
            index = faiss.IndexFlatL2(self.d)
            index.add(array_c)
            D,I = index.search(array_c, 2)
            nn.extend([D[j,1] for j in range(self.k)]) # l2 distance to nearest centroid
            
            # max_dist not meaningful for product quantizer, not using to weight patchiness anyway
            # #add l2 dist from centroid to farthest points
            # for t in range(self.k):
            #     #TODO: why would there be a centroid with no points assigned to it?
            #     if len(self.all_c[i][t]) > 0:
            #         max_dist.append(np.max([np.linalg.norm(self.c[i][t]-j) for j in self.all_c[i][t]])) 
        
        #normalize both lists of distances
        nn /= np.max(nn)
        #max_dist_norm = max_dist/np.max(max_dist)
        
        return nn, #max_dist_norm, #max_dist
    
    
    
    def compare_points(self):
        #considering distribution of point counts over centroids
        #returns var of odds and kl of point proportions (compared to uniform), averaged over subspaces
        
        #sample from uniform to compare in kl_div
        u = np.random.uniform(0, self.k, self.n)
        u_hist = np.histogram(u, bins = self.k)[0]
        u_hist = u_hist/self.n
        
        metrics = []
        all_hist_norm = []
        all_hist = []
        for i in range(self.M):
            h = np.histogram(self.trans[i], bins = self.k)[0]
            h_norm = h/self.n #normalize cell counts
            odds_r = (h_norm/(1-h_norm))*(self.k-1) #use odds-ratio w/ uniform instead of prob to make value more human-readable
            var = np.var(odds_r)
            kl = sum(kl_div(h_norm, u_hist)) #elementwise function needs to be summed
            metrics.append((var,kl))
            all_hist_norm.extend(h_norm)
            all_hist.extend(h)
            
        #report average var and kl over subspaces
        avg_var = np.mean([k[0] for k in metrics])
        avg_kl = np.mean([k[1] for k in metrics])  
        
        return all_hist, all_hist_norm, avg_var, avg_kl
    
    
    def compare_point_dist(self):
        # compare distribution of points assigned to each centroid
        # returns list of EEE scores per centroid
        
        all_EEE = []
        
        for i in range(self.M):
            e = [EEE(j) for j in self.all_c[i] if len(j) > 1] #EEE not meaningful for only one datapoint
            all_EEE.extend(e)
            
        return all_EEE
    
    def patchiness(self, counts):
        # returns adjusted patchiness index from ecology
        # weight the densities to adjust for diff. size cells
        
        m = sum(counts)/(self.k) #avg number points per cell (density)
        V = np.var(counts) #variance of cell densities
        m_star = m +((V/m)-1) #variance adjusted density
        
        return m_star/m #patchiness index
        