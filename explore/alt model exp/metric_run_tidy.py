#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:32:34 2024

@author: anna
"""

import pickle
from datetime import datetime
import os
from math import e
#%%
pre = "/media/anna/Samsung_T5/manifolds/"
suf = "samplespace_20Mar24.pkl"

model_samples = dict()

for filename in os.listdir(pre):
    if filename.endswith(suf):
        m = filename[:-(len(suf))]
        model = pickle.load(open(pre+filename, "rb"))
        model_samples[m] = model

#compare three different samples of Zhang model on metrics for confidence in size
# zhang_sample1 = pickle.load(open("/media/anna/Samsung_T5/manifolds/zhang1_samplespace_14Mar24.pkl", "rb"))
# zhang_sample2 = pickle.load(open("/media/anna/Samsung_T5/manifolds/zhang1_samplespace_18Mar24.pkl", "rb"))
# zhang_sample3 = pickle.load(open("/media/anna/Samsung_T5/manifolds/zhang1_samplespace_2_18Mar24.pkl", "rb"))

# model_samples = {"zhang1" : zhang_sample1,
#               "zhang2" : zhang_sample2,
#               "zhang3" : zhang_sample3}
#%%

for m in model_samples:
    print("model", m)
    data = np.array(model_samples[m])
    filename = m +"_quant_results_" +datetime.today().strftime("%d%b%-y") + ".csv"
    with open(filename, "w") as f:
        f.write("\t".join(["quant", 
                            "error", 
                            "pt_var", 
                            "pt_kl", 
                            "cent_var",
                            "EEE_mean", 
                            "patch",
                            ])+"\n")
     
    print("making pq")
    pq = faiss.ProductQuantizer(768,4,8) #(dim, M subspaces, nbits=256 centroids)
    pq.train(np.array(data))

    print("making aq")
    #additive/local search quantizer
    aq = faiss.LocalSearchQuantizer(768,4,8)
    aq.train(np.array(data))
    
    qs = [pq,aq]
    q_name = ["product", "additive"]
    
    for idx, q in enumerate(qs):
        quant = quantization(data, q)
        print("metrics for", q_name[idx])
        print("error")
        error = quant.avg_reconstruction_error()
        print("centroids")
        cent_var = np.var(quant.compare_centroids())
        print("point counts")
        hist, hist_norm, pt_var, pt_kl = quant.compare_points()
        print("point dist")
        avg_EEE = np.mean(quant.compare_point_dist())
        print("patchiness")
        patch = quant.patchiness(hist)
        
        with open(filename, "a") as f:
            metrics = [error,
                        pt_var, 
                        pt_kl, 
                        cent_var,
                        avg_EEE, 
                        patch,
                        ]
            line = q_name[idx]+"\t"+"\t".join([str(i) for i in metrics])+"\n"
            f.write(line)
            
#%%
for m in model_samples:
    print("model", m)
    data = np.array(model_samples[m])
    filename = m +"_spread_results_" +datetime.today().strftime("%d%b%-y") + ".csv"
    with open(filename, "w") as f:
        f.write("\t".join(["EEE",
                           "VRM",
                           "IsoScore",
                            ])+"\n")
    print("EEE")    
    cumsum = EEE(data)
    print("VRM")
    vas = VRM(data)
    print("IsoScore")
    iso = IsoScore(data)
    
    with open(filename, "a") as f:
        metrics = [cumsum,
                   vas,
                   iso,
                    ]
        line = "\t".join([str(i) for i in metrics])+"\n"
        f.write(line)