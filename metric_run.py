#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:43:14 2024

@author: anna
"""
import pickle
from datetime import datetime

#%%
#model_samples = pickle.load(open("/media/anna/Samsung_T5/manifolds/all_samplespace_22Dec23.pkl", "rb"))

#compare three different samples of Zhang model on metrics for confidence in size
zhang_sample1 = pickle.load(open("/media/anna/Samsung_T5/manifolds/zhang_samplespace_22Dec23.pkl", "rb"))
zhang_sample2 = pickle.load(open("/media/anna/Samsung_T5/manifolds/zhang_samplespace_15Jan24.pkl", "rb"))
zhang_sample3 = pickle.load(open("/media/anna/Samsung_T5/manifolds/zhang_samplespace_18Jan24_2.pkl", "rb"))

model_samples = {"zhang1" : zhang_sample1,
              "zhang2" : zhang_sample2,
              "zhang3" : zhang_sample3}


#%%
#quantization metrics

for m in model_samples:
    print("model", m)
    data = np.array(model_samples[m])
    filename = m +"_quant_results_" +datetime.today().strftime("%d%b%-y") + ".csv"
    # with open(filename, "w") as f:
    #     f.write("\t".join(["quant", 
    #                        "error", 
    #                        #"recall", 
    #                        "pt_var", 
    #                        "pt_kl", 
    #                        "cent_var",
    #                        "far_var",
    #                        "EEE_mean", 
    #                        "patch"])+"\n")
        
    print("making pq")
    pq = faiss.ProductQuantizer(768,4,8) #(dim, M subspaces, nbits=256 centroids)
    pq.train(np.array(data))

    print("making aq")
    #additive/local search quantizer
    aq = faiss.LocalSearchQuantizer(768,4,8)
    aq.train(np.array(data))
    
    qs = [pq,aq]
    q_name = ["product", "additive"]
    
    #EEE_dict = dict()
    for idx, q in enumerate(qs):
        print("metrics for", q_name[idx])
        print("error")
        error = avg_reconstruction_error(data, q)
        # nn search too slow; consider using smaller k?
        # print("recall")
        # recall = NN_recall(data, q, 10, 10)
        print("point_counts")
        c, pt_var, pt_kl = compare_points(data, q)
        print("centroid dist")
        cent_dist, cent_var, far_dist, far_var = compare_centroids(data,q)
        print("cluster dist")
        cent_EEE, EEE_mean = compare_point_dist(data,q)
        print("patchiness")
        patch = patchiness(data,q,c,far_dist)
        
        # EEE_dict[m+q_name[idx]] = (cent_EEE,EEE_mean)
    
        with open(filename, "a") as f:
            metrics = [error,
                        #recall,
                        pt_var, 
                        pt_kl, 
                        cent_var,
                        far_var,
                        EEE_mean, 
                        patch]
            line = q_name[idx]+"\t"+"\t".join([str(i) for i in metrics])+"\n"
            f.write(line)


#%%
#spread metrics

for m in model_samples:
    print("model", m)
    data = model_samples[m]
    filename = m +"_spread_results_" +datetime.today().strftime("%d%b%-y") + ".csv"
    with open(filename, "w") as f:
        f.write("\t".join(["EEE", "VRM", "IsoScore"])+"\n")
    
    print("EEE")
    e = EEE(data)
    print("VRM")
    v = VRM(data)
    print("IsoScore")
    I = IsoScore(data)
    
    with open(filename, "w") as f:
        metrics = [e,v,I]
        line = "\t".join([str(i) for i in metrics])+"\n"
        f.write(line)
