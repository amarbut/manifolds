#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:39:10 2024

@author: anna
"""

from torch import nn, FloatTensor
from transformers import (BertForMaskedLM,
                         RobertaForMaskedLM,
                         BertConfig,
                         BertModel)
from fairseq.models.roberta import RobertaModel
import random
import argparse 
import pickle
import numpy as np
import os

model_dict = {"shuffle_sent":'/media/anna/Samsung_T5/manifolds/roberta.base.shuffle.n1',
              "shuffle_corp":'roberta.base.shuffle.corpus',
              "roberta_sinha":'roberta.base.orig',
              "asci":"aajrami/bert-ascii-base",
              "rand":"aajrami/bert-rand-base",
              "fc":"aajrami/bert-fc-base",
              "mlm":"aajrami/bert-mlm-base",
              "germ":"dbmdz/bert-base-german-uncased",
              "chin":"bert-base-chinese",
              "shuffle_index":"bert-base-uncased"}

def load_alt_model(paper, model_name, save_loc):
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
            model.set_input_embeddings(nn.Embedding.from_pretrained(FloatTensor(shuff_emb)))
    
    filename = save_loc+model_name+".pkl"
    os.makedirs(args["save_loc"], exist_ok = True)
    pickle.dump(model, open(filename, "wb"))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help = 'shorthand name of alternative model', required = True)
    parser.add_argument('--paper', help = 'sinha, aajrami, or zhang', required = True)
    parser.add_argument('--save_loc', help = 'location for saving loaded model', default = '', required = False)
    
    args = vars(parser.parse_args())
    m_dict = load_alt_model(**args)
    







    
