#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:26:20 2024

@author: anna
"""

import os
import numpy as np
import argparse
import json
import pickle
import scipy.stats as stats



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_loc', help = 'Folder holding metric files', default = '', required = False)
    args = vars(parser.parse_args())
    
    folder_loc = args["folder_loc"]
    
    for m in os.listdir(folder_loc):
        print(folder_loc, m)
        metric_file = folder_loc+"/"+m
        ss = np.array(pickle.load(open(metric_file +"/sample_space.pkl", "rb")))
        metrics = json.load(open(metric_file+"/metrics.json", "r"))
        
        aquant_file = "/media/anna/Samsung_T5/manifolds/aquants/"+m
        a_recon = np.array(pickle.load(open(aquant_file+"/recon.pkl", "rb")))
        a_error = np.linalg.norm(ss - a_recon, axis = 1)
        a_error_n = a_error/np.max(a_error)
        
        a_mu = np.mean(a_error)
        a_under_mu = np.sum([1 for i in a_error if i < a_mu])/len(a_error)
        a_under_mu2 = np.sum([1 for i in a_error if i < (a_mu/2)])/len(a_error)
        a_skew = stats.skew(a_error)
        aq75,aq25 = np.percentile(a_error_n, [75,25])
        a_iqr = aq75 - aq25
        
        
        metrics["additive"]["under_mu"] = str(a_under_mu)
        metrics["additive"]["under_mu2"] = str(a_under_mu2)
        metrics["additive"]["skew"] = str(a_skew)
        metrics["additive"]["iqr"] = str(a_iqr)
        
        pquant_file = "/media/anna/Samsung_T5/manifolds/pquants/"+m
        p_recon = np.array(pickle.load(open(pquant_file+"/recon.pkl", "rb")))
        p_error = np.linalg.norm(ss - p_recon, axis = 1)
        p_error_n = p_error/np.max(p_error)
        
        p_mu = np.mean(p_error)
        p_under_mu = np.sum([1 for i in p_error if i < p_mu])/len(p_error)
        p_under_mu2 = np.sum([1 for i in p_error if i < (p_mu/2)])/len(p_error)
        p_skew = stats.skew(p_error)
        pq75,pq25 = np.percentile(p_error_n, [75,25])
        p_iqr = pq75 - pq25
        
        metrics["product"]["under_mu"] = str(p_under_mu)
        metrics["product"]["under_mu2"] = str(p_under_mu2)
        metrics["product"]["skew"] = str(p_skew)
        metrics["product"]["iqr"] = str(p_iqr)
        
        json.dump(metrics, open(metric_file+"/metrics.json", "w"))
        
        
        