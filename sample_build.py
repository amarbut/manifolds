#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:11:16 2023

@author: anna
"""


from torchtext import datasets, data
from torch import nn, FloatTensor
import torch
import transformers
from transformers import BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, BertTokenizer, BertConfig, BertModel
from fairseq.models.roberta import RobertaModel
import random
from argparse import Namespace
from datetime import datetime
import pickle

#%%
#pull PTB and WikiText2 datasets for sampling
train, val, test = datasets.PennTreebank()
t,v,t2 = datasets.WikiText2()

data = [train,val,test,t,v,t2]

#%%
#import sinha models

shuffle_sent = RobertaModel.from_pretrained('roberta.base.shuffle.n1', checkpoint_file='model.pt')
#shuffle_sent.eval()
shuffle_corp = RobertaModel.from_pretrained('roberta.base.shuffle.corpus', checkpoint_file='model.pt')
#shuffle_corp.eval()
roberta_sinha = RobertaModel.from_pretrained('roberta.base.orig', checkpoint_file='model.pt')
#roberta_sinha.eval()

#%%
#import alajrami models

asci = RobertaForMaskedLM.from_pretrained("aajrami/bert-ascii-base")
rand = RobertaForMaskedLM.from_pretrained("aajrami/bert-rand-base")
fc = RobertaForMaskedLM.from_pretrained("aajrami/bert-fc-base")
mlm = RobertaForMaskedLM.from_pretrained("aajrami/bert-mlm-base")
roberta = RobertaForMaskedLM.from_pretrained("roberta-base")

alaj_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

#%%
#build out Zhang models from HF

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

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#%%

def transformer_sample(datasets:list, model, model_source, tokenizer = None, sample_size = 1000):
    emb = []
    
    for i in range(sample_size):
        if i%100 == 0:
            print("sequence", i)
        ds = random.choice(data)
        ds = ds.shuffle()
        sent = list(ds)[0]
        
        if model_source == 'fairseq':
            max_len = model.cfg['model'].max_positions
            inp = model.encode(sent)
            inp = inp[:max_len-3] #adjust for indexing and CLS and SEP tokens
            outp = model.extract_features(inp)
            for v in outp[0]:
                emb.append(v.detach().numpy())
                
        elif model_source == 'huggingface':
            max_len = model.config.max_position_embeddings
            inp = tokenizer(sent, return_tensors = "pt")
            inp['input_ids'] = inp['input_ids'][None,0,:max_len-3] #adjust for indexing and CLS and SEP tokens
            if len(inp.get('token_type_ids')) > 0:
                inp['token_type_ids'] = inp['token_type_ids'][None,0,:max_len-3]
            inp['attention_mask'] = inp['attention_mask'][None,0,:max_len-3]
            outp = model(**inp, output_hidden_states = True)
            for v in outp.hidden_states[12][0]:
                emb.append(v.detach().numpy())
            
    return emb

#%%
#specify model details

model_dict = {#'alaj':{'model':mlm,
              #       'source': 'huggingface',
              #       'tokenizer': alaj_tokenizer},
              # 'ascii':{'model':asci,
              #          'source': 'huggingface',
              #          'tokenizer': alaj_tokenizer},
              # 'fc':{'model':fc,
              #       'source': 'huggingface',
              #       'tokenizer': alaj_tokenizer},
              # 'rand':{'model':rand,
              #         'source': 'huggingface',
              #         'tokenizer': alaj_tokenizer},
              # 'sinha':{'model':roberta_sinha,
              #          'source': 'fairseq',
              #          'tokenizer': None},
              # 'shuffle_sent':{'model':shuffle_sent,
              #                 'source': 'fairseq',
              #                 'tokenizer': None},
              # 'shuffle_corp':{'model':shuffle_corp,
              #                 'source': 'fairseq',
              #                 'tokenizer': None},
              # 'zhang':{'model':bert,
              #           'source': 'huggingface',
              #           'tokenizer': bert_tokenizer},
              # 'germ':{'model':germ,
              #         'source': 'huggingface',
              #         'tokenizer': bert_tokenizer},
              # 'chin':{'model':chin,
              #         'source': 'huggingface',
              #         'tokenizer': bert_tokenizer},
              'untrained':{'model':newMod,
                           'source': 'huggingface',
                           'tokenizer': bert_tokenizer},
              'zhang_shuff': {'model':shuffle,
                        'source': 'huggingface',
                        'tokenizer': bert_tokenizer}
              }

#%%
#build out sample spaces
for m in model_dict:
    print(datetime.now())
    print(m)
    model = model_dict[m]['model']
    model_source = model_dict[m]['source']
    tokenizer = model_dict[m]['tokenizer']
    emb = transformer_sample(data, model, model_source, tokenizer, sample_size = 5000)
    model_dict[m]['sample'] = emb
    filename = m + "_samplespace_" + datetime.today().strftime("%d%b%-y") + ".pkl"
    pickle.dump(emb, open(filename, "wb"))

#moved sample files to ssd

model_samples = dict()

for m in model_dict:
    d = model_dict[m]['sample']
    model_samples[m] = d    
    

fname = "all_samplespace_"+ datetime.today().strftime("%d%b%-y")+".pkl"
pickle.dump(model_samples, open(fname, "wb"))
