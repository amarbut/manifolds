#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:43:52 2024

@author: anna
"""

import os
import pandas as pd
import pickle
import faiss
import numpy as np

metric_dirs = ["/media/anna/Samsung_T5/manifolds/VM_results/VM_1/test/metric_results/prajjwal1/",
               "/media/anna/Samsung_T5/manifolds/VM_results/VM_2/test/metric_results/prajjwal1/",
               "/media/anna/Samsung_T5/manifolds/VM_results/VM_3/test/metric_results/prajjwal1/",
               "/media/anna/Samsung_T5/manifolds/VM_results/VM_4/test/metric_results/prajjwal1/",
               "/media/anna/Samsung_T5/manifolds/VM_results/VM_5/test/metric_results/prajjwal1/",
               "/media/anna/Samsung_T5/manifolds/VM_results/VM_6/test/metric_results/prajjwal1/",
               "/media/anna/Samsung_T5/manifolds/VM_results/VM_7/test/metric_results/prajjwal1/",
               "/media/anna/Samsung_T5/manifolds/VM_results/VM_8/test/metric_results/prajjwal1/",
               "/media/anna/Samsung_T5/manifolds/VM_results/VM_9/test/metric_results/prajjwal1/",
               "/media/anna/Samsung_T5/manifolds/VM_results/VM_10/test/metric_results/prajjwal1/",
               "/media/anna/Samsung_T5/manifolds/float_runs/metric_results/prajjwal1/",
               "/media/anna/Samsung_T5/manifolds/base_compare/metric_results/",
               "/media/anna/Samsung_T5/manifolds/alt_models/"]

new_point_dist = pd.DataFrame(columns = ["p-point_dist", "a-point_dist"])

for d in metric_dirs:
    for model in os.listdir(d):
        fname = d+model+"/sample_space.pkl"
        sample_space = pickle.load(open(fname, "rb"))
        
        data = np.array(sample_space)
        d = len(data[0]) #dimension of embeddings
        
        print("making pq")
        pq = faiss.ProductQuantizer(d,4,8) #(dim, M subspaces, nbits=256 centroids)
        pq.train(np.array(data))
        q1 = quantization(data,pq)
        pe = np.mean(q1.compare_point_dist())
        
        print("making aq")
        #additive/local search quantizer
        aq = faiss.LocalSearchQuantizer(d,4,8)
        aq.train(np.array(data))
        q2 = quantization(data,aq)
        ae = np.mean(q2.compare_point_dist())
        
        df = pd.DataFrame({"p-point_dist":pe,
                           "a-point_dist":ae},
                          index = model)
        new_point_dist = pd.concat([new_point_dist, df])

new_point_dist.to_csv("/media/anna/Samsung_T5/manifolds/new_point_dist.csv")   
        
        