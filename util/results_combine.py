#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:54:07 2024

@author: anna
"""

import os
import json
import pandas as pd

start = "/media/anna/Samsung_T5/manifolds/VM_results"
middle = "/test/glue_results/prajjwal1"

glue = pd.DataFrame(columns = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"])

for VM in os.listdir(start):
    for model in os.listdir(start+"/"+VM+middle):
        for filename in os.listdir(start+"/"+VM+middle+"/"+model):
            if filename == "all_results.json":
                fileloc = start+"/"+VM+middle+"/"+model+"/"+filename
                with open(fileloc, "r") as f:
                    r = json.load(f)
                    df = pd.DataFrame(r, index = [model])
                    glue = pd.concat([glue, df])
                    
glue.to_csv("VM_combined_glue.csv")                 


metric_middle = "/test/metric_results/prajjwal1"

metrics = pd.DataFrame(columns = ["p-error", "p-centroids", "p-point_counts", "p-point_dist", "p-patchiness", "p-iqr",
                                  "a-error", "a-centroids", "a-point_counts", "a-point_dist", "a-patchiness", "a-iqr",
                                  "EEE", "VRM", "IsoScore"])
for VM in os.listdir(start):
    for model in os.listdir(start+"/"+VM+metric_middle):
        for filename in os.listdir(start+"/"+VM+metric_middle+"/"+model):
            if filename == "metrics.json":
                fileloc = start+"/"+VM+metric_middle+"/"+model+"/"+filename
                with open(fileloc, "r") as f:
                    r = json.load(f)
                    df1 = pd.DataFrame(r["product"], index = [model])
                    df1 = df1.rename(columns = {"error":"p-error",
                                          "centroids":"p-centroids",
                                          "point_counts":"p-point_counts",
                                          "point_dist":"p-point_dist",
                                          "patchiness":"p-patchiness",
                                          "iqr": "p-iqr",
                                          "under_mu":"p-under_mu",
                                          "under_mu2":"p-under_mu2",
                                          "skew":"p-skew"})
                    df2 = pd.DataFrame(r["additive"], index = [model])
                    df2 = df2.rename(columns = {"error":"a-error",
                                          "centroids":"a-centroids",
                                          "point_counts":"a-point_counts",
                                          "point_dist":"a-point_dist",
                                          "patchiness":"a-patchiness",
                                          "iqr":"a-iqr",
                                          "under_mu":"a-under_mu",
                                          "under_mu2":"a-under_mu2",
                                          "skew":"a-skew"})
                    df3 = pd.DataFrame(r["spread"], index = [model])
                    merged = pd.concat([df1,df2,df3], axis = 1)
                    metrics = pd.concat([metrics, merged])
                    

metrics[["p-count_var", "p-count_kl"]] = pd.DataFrame(metrics['p-point_counts'].str[1:-1].str.split(",", expand = True), index = metrics.index)
metrics[["a-count_var", "a-count_kl"]] = pd.DataFrame(metrics['a-point_counts'].str[1:-1].str.split(",", expand = True), index = metrics.index)
metrics["IsoScore"] = metrics["IsoScore"].str.replace("tensor(","").str[:-1]
metrics = metrics.drop(["p-point_counts","a-point_counts"] , axis = 1)
metrics = metrics.astype(float)

metrics.to_csv("VM_combined_metrics.csv")

#%%
#add lower alpha data to VM data

glue = pd.read_csv("VM_combined_glue.csv", index_col=0)
metrics = pd.read_csv("VM_combined_metrics.csv", index_col=0)


start = "/media/anna/Samsung_T5/manifolds/float_runs/glue_results/prajjwal1"


for model in os.listdir(start):
    for filename in os.listdir(start+"/"+model):
        if filename == "all_results.json":
            fileloc = start+"/"+model+"/"+filename
            with open(fileloc, "r") as f:
                r = json.load(f)
                df = pd.DataFrame(r, index = [model])
                glue = pd.concat([glue, df])
                    
glue.to_csv("VM_combined_glue_2.csv")                 


metric_start = "/media/anna/Samsung_T5/manifolds/float_runs/metric_results/prajjwal1"

new_metrics = pd.DataFrame(columns = ["p-error", "p-centroids", "p-point_counts", "p-point_dist", "p-patchiness", "p-iqr",
                                  "a-error", "a-centroids", "a-point_counts", "a-point_dist", "a-patchiness", "a-iqr",
                                  "EEE", "VRM", "IsoScore"])

for model in os.listdir(metric_start):
        for filename in os.listdir(metric_start+"/"+model):
            if filename == "metrics.json":
                fileloc = metric_start+"/"+model+"/"+filename
                with open(fileloc, "r") as f:
                    r = json.load(f)
                    df1 = pd.DataFrame(r["product"], index = [model])
                    df1 = df1.rename(columns = {"error":"p-error",
                                          "centroids":"p-centroids",
                                          "point_counts":"p-point_counts",
                                          "point_dist":"p-point_dist",
                                          "patchiness":"p-patchiness",
                                          "iqr":"p-iqr",
                                          "under_mu":"p-under_mu",
                                          "under_mu2":"p-under_mu2",
                                          "skew":"p-skew"})
                    df2 = pd.DataFrame(r["additive"], index = [model])
                    df2 = df2.rename(columns = {"error":"a-error",
                                          "centroids":"a-centroids",
                                          "point_counts":"a-point_counts",
                                          "point_dist":"a-point_dist",
                                          "patchiness":"a-patchiness",
                                          "iqr":"a-iqr",
                                          "under_mu":"a-under_mu",
                                          "under_mu2":"a-under_mu2",
                                          "skew":"a-skew"})
                    df3 = pd.DataFrame(r["spread"], index = [model])
                    merged = pd.concat([df1,df2,df3], axis = 1)
                    new_metrics = pd.concat([new_metrics, merged])
                    

new_metrics[["p-count_var", "p-count_kl"]] = pd.DataFrame(new_metrics['p-point_counts'].str[1:-1].str.split(",", expand = True), index = new_metrics.index)
new_metrics[["a-count_var", "a-count_kl"]] = pd.DataFrame(new_metrics['a-point_counts'].str[1:-1].str.split(",", expand = True), index = new_metrics.index)
new_metrics["IsoScore"] = new_metrics["IsoScore"].str.replace("tensor(","").str[:-1]
new_metrics = new_metrics.drop(["p-point_counts","a-point_counts"] , axis = 1)
new_metrics = new_metrics.astype(float)

metrics = pd.concat([metrics, new_metrics])

metrics.to_csv("VM_combined_metrics_2.csv")

#%%
#load new alt model, metrics

alt_metric_start = "/media/anna/Samsung_T5/manifolds/alt_models/"

alt_metrics = pd.DataFrame(columns = ["p-error", "p-centroids", "p-point_counts", "p-point_dist", "p-patchiness", "p-iqr",
                                  "a-error", "a-centroids", "a-point_counts", "a-point_dist", "a-patchiness", "a-iqr",
                                  "EEE", "VRM", "IsoScore"])


for model in os.listdir(alt_metric_start):
    fname = alt_metric_start+model+"/metrics.json"
    r = json.load(open(fname))
    df1 = pd.DataFrame(r["product"], index = [model])
    df1 = df1.rename(columns = {"error":"p-error",
                          "centroids":"p-centroids",
                          "point_counts":"p-point_counts",
                          "point_dist":"p-point_dist",
                          "patchiness":"p-patchiness",
                          "iqr":"p-iqr",
                          "under_mu":"p-under_mu",
                          "under_mu2":"p-under_mu2",
                          "skew":"p-skew"})
    df2 = pd.DataFrame(r["additive"], index = [model])
    df2 = df2.rename(columns = {"error":"a-error",
                          "centroids":"a-centroids",
                          "point_counts":"a-point_counts",
                          "point_dist":"a-point_dist",
                          "patchiness":"a-patchiness",
                          "iqr":"a-iqr",
                          "under_mu":"a-under_mu",
                          "under_mu2":"a-under_mu2",
                          "skew":"a-skew"})
    df3 = pd.DataFrame(r["spread"], index = [model])
    merged = pd.concat([df1,df2,df3], axis = 1)
    alt_metrics = pd.concat([alt_metrics, merged])
    
alt_metrics[["p-count_var", "p-count_kl"]] = pd.DataFrame(alt_metrics['p-point_counts'].str[1:].str[:-1].str.split(",", expand = True), index = alt_metrics.index)
alt_metrics[["a-count_var", "a-count_kl"]] = pd.DataFrame(alt_metrics['a-point_counts'].str[1:].str[:-1].str.split(",", expand = True), index = alt_metrics.index)
alt_metrics["IsoScore"] = alt_metrics["IsoScore"].str.replace("tensor(","").str[:-1]
alt_metrics = alt_metrics.drop(["p-point_counts","a-point_counts"] , axis = 1)
alt_metrics = alt_metrics.astype(float)

alt_metrics = alt_metrics.join(alt_models[["glue_average"]])

alt_metrics.at['bert-small',"glue_average"]=0.78
alt_metrics.at['mlm',"glue_average"]=0.798
alt_metrics.at['untrained_w_emb', "glue_average"]=0.561

alt_metrics["model"] = alt_metrics.index
# merged["model"] = ["bert-small"]

# alt_models = pd.concat([alt_models, merged])

#%%
#import bert-base models


start = "/media/anna/Samsung_T5/manifolds/base_compare/glue_results"

base_glue = pd.DataFrame(columns = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"])

for model in os.listdir(start):
    if model.find("bert-base-uncased")==0:
        for filename in os.listdir(start+"/"+model):
            if filename == "all_results.json":
                fileloc = start+"/"+model+"/"+filename
                with open(fileloc, "r") as f:
                    r = json.load(f)
                    df = pd.DataFrame(r, index = [model])
                    base_glue = pd.concat([base_glue, df])

base_glue["new_average"] = base_glue.drop(["wnli", "average"], axis = 1).mean(axis = 1)

                    
base_glue.to_csv("bert_base_glue.csv")                 


metric_start = "/media/anna/Samsung_T5/manifolds/base_compare/metric_results"

base_metrics = pd.DataFrame(columns = ["p-error", "p-centroids", "p-point_counts", "p-point_dist", "p-patchiness", "p-iqr",
                                  "a-error", "a-centroids", "a-point_counts", "a-point_dist", "a-patchiness", "a-iqr",
                                  "EEE", "VRM", "IsoScore"])

for model in os.listdir(metric_start):
    if model.find("bert-base-uncased")==0:
        for filename in os.listdir(metric_start+"/"+model):
            if filename == "metrics.json":
                fileloc = metric_start+"/"+model+"/"+filename
                with open(fileloc, "r") as f:
                    r = json.load(f)
                    df1 = pd.DataFrame(r["product"], index = [model])
                    df1 = df1.rename(columns = {"error":"p-error",
                                          "centroids":"p-centroids",
                                          "point_counts":"p-point_counts",
                                          "point_dist":"p-point_dist",
                                          "patchiness":"p-patchiness",
                                          "iqr":"p-iqr",
                                          "under_mu":"p-under_mu",
                                          "under_mu2":"p-under_mu2",
                                          "skew":"p-skew"})
                    df2 = pd.DataFrame(r["additive"], index = [model])
                    df2 = df2.rename(columns = {"error":"a-error",
                                          "centroids":"a-centroids",
                                          "point_counts":"a-point_counts",
                                          "point_dist":"a-point_dist",
                                          "patchiness":"a-patchiness",
                                          "iqr":"a-iqr",
                                          "under_mu":"a-under_mu",
                                          "under_mu2":"a-under_mu2",
                                          "skew":"a-skew"})
                    df3 = pd.DataFrame(r["spread"], index = [model])
                    merged = pd.concat([df1,df2,df3], axis = 1)
                    base_metrics = pd.concat([base_metrics, merged])
                    

base_metrics[["p-count_var", "p-count_kl"]] = pd.DataFrame(base_metrics['p-point_counts'].str[1:].str[:-1].str.split(",", expand = True), index = base_metrics.index)
base_metrics[["a-count_var", "a-count_kl"]] = pd.DataFrame(base_metrics['a-point_counts'].str[1:].str[:-1].str.split(",", expand = True), index = base_metrics.index)
base_metrics["IsoScore"] = base_metrics["IsoScore"].str.replace("tensor(","").str[:-1]
base_metrics = base_metrics.drop(["p-point_counts","a-point_counts"] , axis = 1)
base_metrics = base_metrics.astype(float)
base_metrics = pd.concat([base_metrics, base_glue[["average"]]], axis = 1)
base_metrics = base_metrics.reset_index(names = "model")
base_metrics["alpha"] = base_metrics["model"].str.replace("bert-base-uncased_", "").str[:-2].astype(float)
base_metrics = base_metrics.set_index("model")
base_metrics.to_csv("bert_base_metrics.csv")

#%%
#import roberta-base models


start = "/media/anna/Samsung_T5/manifolds/base_compare/glue_results"

roberta_glue = pd.DataFrame(columns = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"])

for model in os.listdir(start):
    if model.find("roberta-base")==0:
        for filename in os.listdir(start+"/"+model):
            if filename == "all_results.json":
                fileloc = start+"/"+model+"/"+filename
                with open(fileloc, "r") as f:
                    r = json.load(f)
                    df = pd.DataFrame(r, index = [model])
                    roberta_glue = pd.concat([roberta_glue, df])

roberta_glue["average"] = roberta_glue.drop(["wnli", "average"], axis = 1).mean(axis = 1)

roberta_glue.to_csv("roberta_glue.csv")                 


metric_start = "/media/anna/Samsung_T5/manifolds/base_compare/metric_results"

roberta_metrics = pd.DataFrame(columns = ["p-error", "p-centroids", "p-point_counts", "p-point_dist", "p-patchiness","p-iqr",
                                  "a-error", "a-centroids", "a-point_counts", "a-point_dist", "a-patchiness", "a-iqr",
                                  "EEE", "VRM", "IsoScore"])

for model in os.listdir(metric_start):
    if model.find("roberta-base_")==0:
        for filename in os.listdir(metric_start+"/"+model):
            if filename == "metrics.json":
                fileloc = metric_start+"/"+model+"/"+filename
                with open(fileloc, "r") as f:
                    r = json.load(f)
                    df1 = pd.DataFrame(r["product"], index = [model])
                    df1 = df1.rename(columns = {"error":"p-error",
                                          "centroids":"p-centroids",
                                          "point_counts":"p-point_counts",
                                          "point_dist":"p-point_dist",
                                          "patchiness":"p-patchiness",
                                          "iqr":"p-iqr",
                                          "under_mu":"p-under_mu",
                                          "under_mu2":"p-under_mu2",
                                          "skew":"p-skew"})
                    df2 = pd.DataFrame(r["additive"], index = [model])
                    df2 = df2.rename(columns = {"error":"a-error",
                                          "centroids":"a-centroids",
                                          "point_counts":"a-point_counts",
                                          "point_dist":"a-point_dist",
                                          "patchiness":"a-patchiness",
                                          "iqr":"a-iqr",
                                          "under_mu":"a-under_mu",
                                          "under_mu2":"a-under_mu2",
                                          "skew":"a-skew"})
                    df3 = pd.DataFrame(r["spread"], index = [model])
                    merged = pd.concat([df1,df2,df3], axis = 1)
                    roberta_metrics = pd.concat([roberta_metrics, merged])
                    

roberta_metrics[["p-count_var", "p-count_kl"]] = pd.DataFrame(roberta_metrics['p-point_counts'].str[1:].str[:-1].str.split(",", expand = True), index = roberta_metrics.index)
roberta_metrics[["a-count_var", "a-count_kl"]] = pd.DataFrame(roberta_metrics['a-point_counts'].str[1:].str[:-1].str.split(",", expand = True), index = roberta_metrics.index)
roberta_metrics["IsoScore"] = roberta_metrics["IsoScore"].str.replace("tensor(","").str[:-1]
roberta_metrics = roberta_metrics.drop(["p-point_counts","a-point_counts"] , axis = 1)
roberta_metrics = roberta_metrics.astype(float)
roberta_metrics = pd.concat([roberta_metrics, roberta_glue[["average"]]], axis = 1)
roberta_metrics = roberta_metrics.reset_index(names = "model")
roberta_metrics["alpha"] = roberta_metrics["model"].str.replace("roberta-base_", "").str[:-2].astype(float)
roberta_metrics = roberta_metrics.set_index("model")
roberta_metrics.to_csv("roberta_metrics.csv")