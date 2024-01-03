#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:44:33 2023

@author: anna
"""

from transformers import BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, BertTokenizer, BertConfig, BertModel
import random
from torchtext import datasets
from fairseq.models.roberta import RobertaModel
from torch import nn, FloatTensor
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from scipy.stats import differential_entropy, multivariate_normal
import numpy as np
from math import e
import matplotlib.pyplot as plt  
import faiss 
from sklearn.metrics import silhouette_score
#from sklearn.cluster import KMeans
from datetime import datetime,date
import pickle

train, val, test = datasets.PennTreebank()

def build_models(model_list):
    model_dict = dict()
    if "alaj" in model_list:
        alaj_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
      
        model_dict['alaj']={'model':RobertaForMaskedLM.from_pretrained("aajrami/bert-mlm-base"),
                          'source': 'huggingface',
                          'tokenizer': alaj_tokenizer} 
        model_dict['ascii']={'model': RobertaForMaskedLM.from_pretrained("aajrami/bert-ascii-base"),
                           'source': 'huggingface',
                           'tokenizer': alaj_tokenizer}
        model_dict['fc']={'model': RobertaForMaskedLM.from_pretrained("aajrami/bert-fc-base"),
                        'source': 'huggingface',
                        'tokenizer': alaj_tokenizer}
        model_dict['rand']={'model':RobertaForMaskedLM.from_pretrained("aajrami/bert-rand-base"),
                          'source': 'huggingface',
                          'tokenizer': alaj_tokenizer}
        
    if 'sinha' in model_list:
        
        shuffle_sent = RobertaModel.from_pretrained('roberta.base.shuffle.n1', checkpoint_file='model.pt')
        shuffle_sent.eval()  # disable dropout (or leave in train mode to finetune)

        shuffle_corp = RobertaModel.from_pretrained('roberta.base.shuffle.corpus', checkpoint_file='model.pt')
        shuffle_corp.eval()

        roberta_sinha = RobertaModel.from_pretrained('roberta.base.orig', checkpoint_file='model.pt')
        roberta_sinha.eval()
        
        model_dict['sinha']={'model':roberta_sinha,
                 'source': 'fairseq',
                 'tokenizer': None}
        model_dict['shuffle_sent']={'model':shuffle_sent,
                        'source': 'fairseq',
                        'tokenizer': None}
        model_dict['shuffle_corp']={'model':shuffle_corp,
                        'source': 'fairseq',
                        'tokenizer': None}
        
    if 'zhang' in model_list:
        bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_emb = bert.get_input_embeddings()

        germ = BertForMaskedLM.from_pretrained("dbmdz/bert-base-german-uncased")
        germ.set_input_embeddings(bert_emb)
        germ.config.vocab_size = 30522

        chin = BertForMaskedLM.from_pretrained("bert-base-chinese")
        chin.set_input_embeddings(bert_emb)
        chin.config.vocab_size = 30522

        #obviously not interesting to visualize--same latent space as original
        shuffle = BertForMaskedLM.from_pretrained("bert-base-uncased")
        shuff_emb = bert_emb.weight.data.numpy()
        random.shuffle(shuff_emb)
        shuffle.set_input_embeddings(nn.Embedding.from_pretrained(FloatTensor(shuff_emb)))

        config = BertConfig()
        newMod = BertModel(config)
        
        model_dict['zhang']={'model':bert,
                  'source': 'huggingface',
                  'tokenizer': bert_tokenizer}
        model_dict['germ']={'model':germ,
                'source': 'huggingface',
                'tokenizer': bert_tokenizer}
        model_dict['chin']={'model':chin,
                'source': 'huggingface',
                'tokenizer': bert_tokenizer}
        model_dict['untrained']={'model':newMod,
                     'source': 'huggingface',
                     'tokenizer': bert_tokenizer}
        model_dict['zhang_shuff']= {'model':shuffle,
                  'source': 'huggingface',
                  'tokenizer': bert_tokenizer}
        
    return model_dict

def transformer_sample(data:list, model, model_source, tokenizer = None, sample_size = 1000):
    emb = []
    
    for i in range(sample_size):
        ds = random.choice(data)
        ds = ds.shuffle()
        sent = list(ds)[0]
        
        if model_source == 'fairseq':
                inp = model.encode(sent)
                outp = model.extract_features(inp)
                for v in outp[0]:
                    emb.append(v.detach().numpy())
                
        elif model_source == 'huggingface':
            inp = tokenizer(sent, return_tensors = "pt")
            outp = model(**inp, output_hidden_states = True)
            for v in outp.hidden_states[12][0]:
                emb.append(v.detach().numpy())
            
    return emb

def build_embeddings(model_dict, data:list):
    model_samples = dict()
    
    for m in model_dict:
        print(datetime.now())
        print(m)
        model = model_dict[m]['model']
        model_source = model_dict[m]['source']
        tokenizer = model_dict[m]['tokenizer']
        emb = transformer_sample(data, model, model_source, tokenizer)
        model_dict[m]['sample'] = emb
        model_samples[m] = emb
        
    store_f = 'ptb_samples_'+date.today().strftime('%d%b%y')+'pkl'
    
    pickle.dump(model_samples, open(store_f))
    
def AUC_eigensum(embeddings, plot = False):
    pc = PCA()
    pc.fit(embeddings)
    num_pc = pc.n_components_
    eigensum = np.cumsum(pc.explained_variance_)
    ref = np.cumsum([eigensum[-1]/num_pc]*num_pc)
    AUC_sum = eigensum-ref
    AUC = auc(range(num_pc),AUC_sum)
    
    total_poss = (eigensum[-1]*num_pc)/2
    
    if plot == True:
        plt.plot(range(num_pc), eigensum, label = 'Cumulative Sum of Eigenvalues')
        plt.plot(range(num_pc), ref)
        plt.legend()
        plt.show()
        
    return round(AUC/total_poss,4)      

def vasicek_entropy(embeddings):
    N = len(embeddings)
    d = len(embeddings[0])
    pc = PCA()
    tx = pc.fit_transform(embeddings)
    
    #fix total explained variance in data to match that of a standard normal distribution
    comp,sample, ev = normal_compare(d,N)
    scale = ev/sum(pc.explained_variance_)
    tx *= np.sqrt(scale)
     
    #compute theoretical max vasicek entropy for std normal comparison
    m = np.log(np.sqrt(2*np.pi*e))
    
    ent = differential_entropy(tx)
    r = np.exp(ent)/np.exp(m)
    
    #report mean SE from ratio of 1
    return round(np.sum((1-r)**2)/d,4)

def normal_compare(dimension, size):
    x = multivariate_normal([0]*dimension, np.identity(dimension))
    y = x.rvs(size = size)
    pc = PCA()
    v = sum(pc.fit(y).explained_variance_)
    return x,y,v

def max_mean_sil(y):
    sil = []
    sil_std = []
    cands = list(range(2,15))
    for k in cands:
        score = []
        kmeans = faiss.Kmeans(d = 768,k=k, gpu = True)
        kmeans.train(y)
        D,labels = kmeans.index.search(np.array(y),1)
        score.append(silhouette_score(y, labels.ravel()))
        sil.append(np.mean(score))
        sil_std.append(np.std(score))
        #print("k&score&std:", k, sil[-1], sil_std[-1])
    if max(sil) >= 0.1:
        best_k = cands[sil.index(max(sil))]
    else:
        best_k = 1
        
    #print('bestk&sil&std:', best_k, max(sil), std)
    return (best_k, max(sil))

def calculate_metrics(model_dict):
    for m in model_dict:
        d = model_dict[m]['sample']
        EEE = AUC_eigensum(d)
        VRM = vasicek_entropy(d)
        best_k, msil = max_mean_sil(d)
        model_dict[m]['metrics'] = {'EEE':EEE,
                                    'VRM':VRM,
                                    'K':best_k,
                                    "MSIL": msil}
        print(m)
        print('EEE:', EEE)
        print('VRM:', VRM)
        print('K:', best_k)
        print('MSIL:', msil)
    

    
    
    



