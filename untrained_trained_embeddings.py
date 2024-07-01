#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:35:57 2024

@author: anna
"""

from torch import nn, FloatTensor
from transformers import (BertForMaskedLM,
                         RobertaForMaskedLM,
                         BertConfig,
                         BertModel)
import random
import argparse 
import pickle
import numpy as np
import os

bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
bert_emb = bert.get_input_embeddings().weight.data.numpy()

config = BertConfig()
model = BertModel(config)
model.set_input_embeddings(nn.Embedding.from_pretrained(FloatTensor(bert_emb)))

