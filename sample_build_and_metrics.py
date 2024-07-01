#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:30:13 2024

@author: anna
"""

import pickle
from transformers import (BertTokenizer,
                          BertModel,
                          RobertaTokenizer,
                          BertForMaskedLM,
                          RobertaForMaskedLM,
                          BertConfig)
import faiss
import numpy as np
from scipy.special import kl_div
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from scipy.stats import differential_entropy, multivariate_normal, skew
from math import e
from IsoScore.IsoScore import *
import json
import argparse
import os
from torch import nn, FloatTensor
from fairseq.models.roberta import RobertaModel
import random


model_dict = {"shuffle_sent":'roberta.base.shuffle.n1',
              "shuffle_corp":'roberta.base.shuffle.corpus',
              "roberta_sinha":'roberta.base.orig',
              "asci":"aajrami/bert-ascii-base",
              "rand":"aajrami/bert-rand-base",
              "fc":"aajrami/bert-fc-base",
              "mlm":"aajrami/bert-mlm-base",
              "germ":"dbmdz/bert-base-german-uncased",
              "chin":"bert-base-chinese",
              "shuffle_index":"bert-base-uncased"}

def load_alt_model(paper, model_name):
    m_loc = model_dict[model_name]
    
    if paper == "sinha":
        model = RobertaModel.from_pretrained(m_loc, checkpoint_file='model.pt')

        
    if paper == "aajrami":
        model = RobertaForMaskedLM.from_pretrained(m_loc)
        
    if paper == "zhang":
        bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
        bert_emb = bert.get_input_embeddings() 
        
        if model_name == "germ" or model_name == "chin":
            model = BertForMaskedLM.from_pretrained(m_loc)
            model.set_input_embeddings(bert_emb)
            model.config.vocab_size = 30522
            
        if model_name == "shuffle_index":
            model = BertForMaskedLM.from_pretrained("bert-base-uncased")
            shuff_emb = np.copy(bert_emb.weight.data.numpy())
            random.shuffle(shuff_emb)
            model.set_input_embeddings(nn.Embedding.from_pretrained(FloatTensor(shuff_emb)))
            
        if model_name == "untrained":
            config = BertConfig()
            model = BertModel(config)
    
    return model

def build_samples(sequences, model, tokenizer = "prajjwal1/bert-small", paper = None, model_type = "bert"):
    # run sample sequences through model to produce sample latent space
    emb = []
    
    #fairseq models run differently
    if paper == "sinha":
        for idx, sent in enumerate(sequences):
            if idx%100 == 0 :
                print("running sentence ", str(idx))
            max_len = model.cfg['model'].max_positions
            inp =  model.encode(sent)
            inp = inp[:max_len-3] #adjust for indexing and CLS and SEP tokens
            outp =  model.extract_features(inp)
            for v in outp[0]:
                emb.append(v.detach().numpy())
    
    #huggingface models require tokenizer            
    else:
        if paper == "aajrami":
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        elif paper == "zhang":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif model_type == "bert":
            tokenizer = BertTokenizer.from_pretrained(tokenizer)
        elif model_type == "roberta":
            tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        
        #find number of layers in model
        layers = model.config.num_hidden_layers
        max_len = model.config.max_position_embeddings
        
        
        for idx, sent in enumerate(sequences):
            if idx%100 == 0 :
                print("running sentence ", str(idx))
            inp = tokenizer(sent, return_tensors = "pt")
            inp['input_ids'] = inp['input_ids'][None,0,:max_len-3] #adjust for indexing and CLS and SEP tokens
            if inp.get('token_type_ids') != None: #Roberta tokenizer doesn't use this
                inp['token_type_ids'] = inp['token_type_ids'][None,0,:max_len-3]
            inp['attention_mask'] = inp['attention_mask'][None,0,:max_len-3]
            outp = model(**inp, output_hidden_states = True)
            for vector in outp.hidden_states[layers][0]:
                emb.append(vector.detach().numpy())
        
    return emb

def normal_compare(dimension, size):
    #sample from a multivariate normal as a reference dist.
    x = multivariate_normal([0]*dimension, np.identity(dimension))
    y = x.rvs(size = size)
    pc = PCA()
    v = sum(pc.fit(y).explained_variance_)
    return x,y,v

def VRM(embeddings):
    N = len(embeddings) #number of embeddings
    d = len(embeddings[0]) #dimension of embeddings
    pc = PCA()
    tx = pc.fit_transform(embeddings) #PCA projection of embeddings
    
    #fix total explained variance in data to match that of a standard normal distribution
    comp,sample, ev = normal_compare(d,N)
    scale = ev/sum(pc.explained_variance_)
    tx *= np.sqrt(scale)
     
    #compute theoretical max vasicek entropy for std normal comparison
    m = np.log(np.sqrt(2*np.pi*e))
    
    ent = differential_entropy(tx)
    r = np.exp(ent)/np.exp(m)
    
    #report mean SSE from ratio of 1
    return round(np.sum((1-r)**2)/d,4)

def EEE(embeddings):
    pc = PCA()
    pc.fit(embeddings)
    num_pc = pc.n_components_ #could also use dimension of embeddings
    
    #cumulative sum of exp. var of true components and uniform reference
    eigensum = np.cumsum(pc.explained_variance_) 
    ref = np.cumsum([eigensum[-1]/num_pc]*num_pc)
    
    #bring true cumulative sum to x axis (subtract diagonal) to find AUC
    AUC_sum = eigensum-ref 
    AUC = auc(range(num_pc),AUC_sum)
    
    total_poss = (eigensum[-1]*num_pc)/2 #total possible AUC
    
    #return AUC as a proportion of the total possible
    return round(AUC/total_poss,4) 

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
        error_dist = np.linalg.norm(self.data - recon, axis = 1)
        error_dist /= np.max(error_dist)
        q75,q25 = np.percentile(error_dist, [75,25])
        iqr = q75 - q25
        return iqr
    
    def reconstruction_skew(self):
        #return skewness of reconstruction error distribution
        recon = self.quantizer.decode(self.codes)
        error_dist = np.linalg.norm(self.data - recon, axis = 1)
        sk = skew(error_dist)
        return sk
    
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
    
    
    def compare_point_dist(self):
        # compare distribution of points assigned to each centroid
        # returns list of EEE scores per centroid
        
        all_EEE = []
        
        for i in range(self.M):
            e = []
            for j in self.all_c[i]:
                if len(j)>1:
                    try:
                        e.append(EEE(j))
                    except:
                        pass
            #EEE not meaningful for only one datapoint
            all_EEE.extend(e)
            
        return all_EEE
    
    def patchiness(self, counts):
        # returns adjusted patchiness index from ecology
        
        m = sum(counts)/(self.k) #avg number points per cell (density)
        V = np.var(counts) #variance of cell densities
        m_star = m +((V/m)-1) #variance adjusted density
        
        return m_star/m #patchiness index

def metric_run(sample_space):
    data = np.array(sample_space)
    d = len(data[0]) #dimension of embeddings
    
    print("making pq")
    pq = faiss.ProductQuantizer(d,4,8) #(dim, M subspaces, nbits=256 centroids)
    pq.train(np.array(data))
    
    print("making aq")
    #additive/local search quantizer
    aq = faiss.LocalSearchQuantizer(d,4,8)
    aq.train(np.array(data))
    
    qs = [pq,aq]
    q_name = ["product", "additive"]
    
    metrics = dict()
    
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
        print("iqr")
        iqr = quant.reconstruction_IQR()
        print("skew")
        sk = quant.reconstruction_skew()
        
        metrics[q_name[idx]] = {"error": str(error),
                            "centroids": str(cent_var),
                            "point_counts":str((pt_var,pt_kl)),
                            "point_dist": str(avg_EEE),
                            "patchiness": str(patch),
                            "iqr": str(iqr),
                            "skew": str(sk)}
    
    print("EEE")    
    cumsum = EEE(data)
    print("VRM")
    vas = VRM(data)
    print("IsoScore")
    iso = IsoScore(data)
    
    metrics["spread"] = {"EEE":str(cumsum),
                         "VRM":str(vas),
                         "IsoScore":str(iso)}
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help = 'Name of pre-trained Bert model or shortname for alt model', default = 'prajjwal1/bert-small', required = False)
    parser.add_argument('--sequences', help = 'Sequences for building out sample space', required = True)
    parser.add_argument('--save_loc', help = 'Location for saving sample space and metrics', default = '', required = False)
    parser.add_argument('--paper', help = 'sinha, aajrami, zhang, or None', default = None, required = False)
    parser.add_argument('--tokenizer', help = 'Name of pre-trained tokenizer', default = 'prajjwal1/bert-small', required = False)
    parser.add_argument('--model_type', help = 'bert or roberta', default = 'bert', required = False)
    args = vars(parser.parse_args())
    
    os.makedirs(args["save_loc"], exist_ok = True)
    
    
    #build out sample latent space and save to file
    sequences = pickle.load(open(args["sequences"], "rb"))
    if args["paper"] == None:
        if args["model_type"] == "bert":
            model = BertModel.from_pretrained(args["model"])
        elif args["model_type"] == "roberta":
            model = RobertaForMaskedLM.from_pretrained(args["model"])
    else:
        model = load_alt_model(model_name = args["model"], paper = args["paper"])
        
    sample_space = build_samples(sequences, model, paper = args["paper"],tokenizer = args["tokenizer"],model_type = args["model_type"])
    space_file = args["save_loc"] +"/sample_space.pkl"
    pickle.dump(sample_space, open(space_file, "wb"))
    
    #run metrics on sample latent space
    metric_dict = metric_run(sample_space)
    metric_file = args["save_loc"]+"/metrics.json"
    json.dump(metric_dict, open(metric_file, "w"))
              
              