#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:18:02 2023

@author: anna
"""

from transformers import BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, BertTokenizer, BertConfig, BertModel
from datasets import load_dataset
import random

ptb = load_dataset("ptb_text_only")

bert_base = BertForMaskedLM.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


#%%
from torchtext import datasets, data
import transformers

train, val, test = datasets.PennTreebank()
t,v,t2 = datasets.WikiText2()

data = [train,val,test,t,v,t2]

#%%
!wget https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.orig.tar.gz
!tar -xzvf roberta.base.orig.tar.gz
# # Copy the dictionary files
# !wget -O dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt && wget -O encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json && wget -O vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
#%%
# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
shuffle_sent = RobertaModel.from_pretrained('roberta.base.shuffle.n1', checkpoint_file='model.pt')
shuffle_sent.eval()  # disable dropout (or leave in train mode to finetune)

shuffle_corp = RobertaModel.from_pretrained('roberta.base.shuffle.corpus', checkpoint_file='model.pt')
shuffle_corp.eval()

roberta_sinha = RobertaModel.from_pretrained('roberta.base.orig', checkpoint_file='model.pt')
roberta_sinha.eval()

# inp = shuffle_sent.encode("This is a sentence")
# outp = shuffle_sent.extract_features(inp)

ptb_outputs = []
data = [test, train, val]

for i in range(1000):
    ds = random.choice(data)
    ds = ds.shuffle()
    sent = list(ds)[0]
    inp = shuffle_corp.encode(sent)
    outp = shuffle_corp.extract_features(inp)
    for v in outp[0]:
        ptb_outputs.append(v.detach().numpy())
#%%
#alajrami models
asci = RobertaForMaskedLM.from_pretrained("aajrami/bert-ascii-base")
rand = RobertaForMaskedLM.from_pretrained("aajrami/bert-rand-base")
fc = RobertaForMaskedLM.from_pretrained("aajrami/bert-fc-base")
mlm = RobertaForMaskedLM.from_pretrained("aajrami/bert-mlm-base")
roberta = RobertaForMaskedLM.from_pretrained("roberta-base")

alaj_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

#%%
#Zhang models
from torch import nn, FloatTensor

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
#%%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


ptb_outputs = []
data = [test, train, val]

for i in range(1000):
    ds = random.choice(data)
    ds = ds.shuffle()
    sent = list(ds)[0]
    inp = bert_tokenizer(sent, return_tensors = "pt")
    outp = newMod(**inp, output_hidden_states = True)
    for v in outp.hidden_states[12][0]:
        ptb_outputs.append(v.detach().numpy())
    
# ptb_outputs = [i.detach().numpy() for i in ptb_outputs]

# i = 0
# for line in ds:
#     if i < 5:
#         print(line)
#     else:
#         break
#     i += 1
        
pca = PCA(n_components = 2).fit(ptb_outputs)
tx = pca.transform(ptb_outputs)

fig = plt.figure(figsize=(4, 3))
plt.scatter(tx[:,0], tx[:,1], s=1, alpha=0.3, marker='.')
plt.title("Untrained BERT Base")
fig.show()
plt.savefig("untrained_ptb.png", format = 'png')

tsne = TSNE(n_components=2).fit_transform(np.array(ptb_outputs))
fig = plt.figure(figsize=(4, 3))
plt.scatter(tsne[:,0], tsne[:,1], s=1, alpha=0.3, marker='.')
plt.title("BERT Base w/ Shuffled Embeddings")
fig.show()
plt.savefig("untrained_ptb_tsne.png", format = 'png')

#%%
#randomly choose sample_size sentences from dataset and pass through model; save as a list of model representations

def transformer_sample(datasets:list, model, model_source, tokenizer = None, sample_size = 1000):
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

#%%
from datetime import datetime

model_dict = {'alaj':{'model':mlm,
                      'source': 'huggingface',
                      'tokenizer': alaj_tokenizer},
              'ascii':{'model':asci,
                       'source': 'huggingface',
                       'tokenizer': alaj_tokenizer},
              'fc':{'model':fc,
                    'source': 'huggingface',
                    'tokenizer': alaj_tokenizer},
              'rand':{'model':rand,
                      'source': 'huggingface',
                      'tokenizer': alaj_tokenizer},
              'sinha':{'model':roberta_sinha,
                       'source': 'fairseq',
                       'tokenizer': None},
              'shuffle_sent':{'model':shuffle_sent,
                              'source': 'fairseq',
                              'tokenizer': None},
              'shuffle_corp':{'model':shuffle_corp,
                              'source': 'fairseq',
                              'tokenizer': None},
              'zhang':{'model':bert,
                        'source': 'huggingface',
                        'tokenizer': bert_tokenizer},
              'germ':{'model':germ,
                      'source': 'huggingface',
                      'tokenizer': bert_tokenizer},
              'chin':{'model':chin,
                      'source': 'huggingface',
                      'tokenizer': bert_tokenizer},
              'untrained':{'model':newMod,
                           'source': 'huggingface',
                           'tokenizer': bert_tokenizer},
              'zhang_shuff': {'model':shuffle,
                        'source': 'huggingface',
                        'tokenizer': bert_tokenizer}
              }
              

for m in model_dict:
    print(datetime.now())
    print(m)
    model = model_dict[m]['model']
    model_source = model_dict[m]['source']
    tokenizer = model_dict[m]['tokenizer']
    emb = transformer_sample(data, model, model_source, tokenizer, sample_size = 5000)
    model_dict[m]['sample'] = emb

model_samples = dict()

for m in model_dict:
    d = model_dict[m]['sample']
    model_samples[m] = d    
    
import pickle

pickle.dump(model_samples, open("ptb_model_samples_21JDec23.pkl", "wb"))



#%%
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from scipy.stats import entropy, differential_entropy, multivariate_normal
from scipy.special import gamma, kl_div, rel_entr
from scipy.spatial import KDTree
import numpy as np
from math import exp,e
import argparse
import matplotlib.pyplot as plt  
import faiss 
from sympy import EulerGamma
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

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
    all_labels = []
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

#%%

for m in model_dict:
    if 'metrics' not in model_dict[m].keys():
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
    
#%%

