#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:41:51 2024

@author: anna
"""

import pickle
import faiss
import numpy as np
from scipy.special import kl_div
import json
import argparse
import os

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
        error = ((self.data - recon)**2).sum() / (self.data ** 2).sum()
        return error
    
    def reconstruction_IQR(self):
        # width of IQR of the normalized reconstruction error distribution
        recon = self.quantizer.decode(self.codes)
        error_dist = np.linalg.norm(self.data - recon)
        error_dist /= np.max(error_dist)
        q75,q25 = np.percentile(error_dist, [75,25])
        iqr = q75 - q25
        return iqr
    
    def compare_centroids(self):
        # considering distribution of centroids
        # returns list of normalized distances to nearest centroid per centroid
        nn = []
        #max_dist = []
        for i in range(self.M):
            # use FAISS L2 nn index to find nearest centroid
            array_c = np.array(self.c[i]).astype('float32')
            index = faiss.IndexFlatL2(self.d)
            index.add(array_c)
            D,I = index.search(array_c, 2)
            nn.extend([D[j,1] for j in range(self.k)]) # l2 distance to nearest centroid
        
        #normalize 
        nn /= np.max(nn)
        
        return nn
    
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
    
    
    # def compare_point_dist(self):
    #     # compare distribution of points assigned to each centroid
    #     # returns list of EEE scores per centroid
        
    #     all_EEE = []
        
    #     for i in range(self.M):
    #         e = [EEE(j) for j in self.all_c[i] if len(j)>1] #EEE not meaningful for only one datapoint
    #         all_EEE.extend(e)
            
    #     return all_EEE
    
    def patchiness(self, counts):
        # returns adjusted patchiness index from ecology
        
        m = sum(counts)/(self.k) #avg number points per cell (density)
        V = np.var(counts) #variance of cell densities
        m_star = m +((V/m)-1) #variance adjusted density
        
        return m_star/m #patchiness index
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_loc', help = 'Folder holding metric files', default = '', required = False)
    args = vars(parser.parse_args())
    
    folder_loc = args["folder_loc"]
    for m in os.listdir(folder_loc):
        print(folder_loc, m)
        metric_file = folder_loc+"/"+m
        ss = np.array(pickle.load(open(metric_file +"/sample_space.pkl", "rb")))
        d = len(ss[0])
        
        print("running pq")
        pq = faiss.ProductQuantizer(d,4,8) #(dim, M subspaces, nbits=256 centroids)
        pq.train(ss) 
        
        pquant = quantization(ss, pq)
        pquant_loc = "/media/anna/Samsung_T5/manifolds/pquants/"
        os.makedirs(pquant_loc+m+"/", exist_ok = True)
        pickle.dump(pquant.c, open(pquant_loc+m+"/codebook.pkl", "wb"))
        pickle.dump(pquant.codes, open(pquant_loc+m+"/codes.pkl", "wb"))
        p_recon = pquant.quantizer.decode(pquant.codes)
        pickle.dump(p_recon, open(pquant_loc+m+"/recon.pkl", "wb"))
        
        print("running aq")
        aq = faiss.LocalSearchQuantizer(d,4,8)
        aq.train(ss) 
        
        aquant = quantization(ss, aq)
        aquant_loc = "/media/anna/Samsung_T5/manifolds/aquants/"
        os.makedirs(aquant_loc+m+"/", exist_ok = True)
        pickle.dump(aquant.c, open(aquant_loc+m+"/codebook.pkl", "wb"))
        pickle.dump(aquant.codes, open(aquant_loc+m+"/codes.pkl", "wb"))
        a_recon = aquant.quantizer.decode(aquant.codes)
        pickle.dump(a_recon, open(aquant_loc+m+"/recon.pkl", "wb"))
                    
    
    