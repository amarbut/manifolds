#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:27:36 2024

@author: anna
"""
m = BertForMaskedLM.from_pretrained("bert-base-uncased")

for name, param in m.named_parameters():
    if name[:7] == "encoder" and name[-6:] == "weight":
        param.requires_grad = False
        noise = (alpha/100)*torch.randn(param.shape) #N(0,1) noise
        param *= (1-(alpha/100))
        param += noise
        param.requires_grad = True
        
#%%
import tensorflow as tf

f = '/media/anna/Samsung_T5/manifolds/base_compare/glue_results/roberta-base_1_1/mnli/runs/May19_23-36-58_anna-4-of-4/events.out.tfevents.1716172446.anna-4-of-4'

for event in tf.compat.v1.train.summary_iterator(f):
    for value in event.summary.value:
        print(value.tag)
        if value.HasField('simple_value'):
            print(value.simple_value)
            
#%%
glue_parse("/media/anna/Samsung_T5/manifolds/base_compare/glue_results/roberta-base_3_1")

#%%

m = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/manifolds/base_compare/roberta-base_10_1")
m2 =RobertaForMaskedLM.from_pretrained("roberta-base")

#%%
#compare two bert-base models (alpha = 5) as sanity check on vals
metrics1 = json.load(open("/media/anna/Samsung_T5/manifolds/base_compare/metric_results/bert-base-uncased_5_1/metrics.json"))
metrics2 = json.load(open("/media/anna/Samsung_T5/manifolds/bert-base-uncased_5_1/metrics.json"))

cols = ["p-error", "p-centroids", "p-point_counts", "p-point_dist", "p-patchiness",
                                  "a-error", "a-centroids", "a-point_counts", "a-point_dist", "a-patchiness",
                                  "EEE", "VRM", "IsoScore"]

compare = pd.DataFrame(columns = ["p-error", "p-centroids", "p-point_counts", "p-point_dist", "p-patchiness",
                                  "a-error", "a-centroids", "a-point_counts", "a-point_dist", "a-patchiness",
                                  "EEE", "VRM", "IsoScore"])
models = ["bert-base_5_1","bert-base_5_2"]

for idx,r in enumerate([metrics1,metrics2]):
    df1 = pd.DataFrame(r["product"], index = [models[idx]])
    df1 = df1.rename(columns = {"error":"p-error",
                          "centroids":"p-centroids",
                          "point_counts":"p-point_counts",
                          "point_dist":"p-point_dist",
                          "patchiness":"p-patchiness"})
    df2 = pd.DataFrame(r["additive"], index = [models[idx]])
    df2 = df2.rename(columns = {"error":"a-error",
                          "centroids":"a-centroids",
                          "point_counts":"a-point_counts",
                          "point_dist":"a-point_dist",
                          "patchiness":"a-patchiness"})
    df3 = pd.DataFrame(r["spread"], index = [models[idx]])
    merged = pd.concat([df1,df2,df3], axis = 1)[cols]
    compare = pd.concat([compare, merged])
    
compare[["p-count_var", "p-count_kl"]] = pd.DataFrame(compare['p-point_counts'].str[1:-1].str.split(",", expand = True), index = compare.index)
compare[["a-count_var", "a-count_kl"]] = pd.DataFrame(compare['a-point_counts'].str[1:-1].str.split(",", expand = True), index = compare.index)
compare["IsoScore"] = compare["IsoScore"].str.replace("tensor(","").str[:-1]
compare = compare.drop(["p-point_counts","a-point_counts"] , axis = 1)
compare = compare.astype(float)

#%%
data = pickle.load(open("/media/anna/Samsung_T5/manifolds/base_compare/metric_results/roberta-base_6.5_1/sample_space.pkl","rb"))

#u_data = np.unique(data, axis = 1) #all vectors are unique

d = len(data[0])
pq = faiss.ProductQuantizer(d,4,8) #(dim, M subspaces, nbits=256 centroids)
pq.train(np.array(data))

quant = quantization(np.array(data), pq)

quant.compare_point_dist() # SVD does not converge error

all_EEE = []

for i in range(quant.M):
    e = []
    for j in quant.all_c[i]:
        if len(j)>1:
            e.append(EEE(j))
    all_EEE.extend(e)
    
#%%
model = RobertaForMaskedLM.from_pretrained("roberta-base")
model.config
