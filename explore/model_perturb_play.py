#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:15:00 2024

@author: anna
"""

from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("prajjwal1/bert-small") 
model2 = BertModel.from_pretrained("prajjwal1/bert-small")
model3 = BertModel.from_pretrained("prajjwal1/bert-small")

#%%
#add noise to all encoder weights
for name, param in m1.named_parameters():
    if name[:7] == "encoder" and name[-6:] == "weight":
        param.requires_grad = False
        noise = torch.randn(param.shape) #N(0,1) noise
        param += noise

#%%
#make sure that in-place noise adjustment actually worked--it does!      
params1 = []
for name, param in m1.named_parameters():
    if name[8:15] == "encoder" and name[-6:] == "weight":
        params1.append(param)
        
params2 = []
for name, param in m2.named_parameters():
    if name[8:15] == "encoder" and name[-6:] == "weight":
        params2.append(param)    
        
params3 = []
for name, param in m3.named_parameters():
    if name[8:15] == "encoder" and name[-6:] == "weight":
        params3.append(param) 
        
[params1[i]==params3[i] for i in range(len(params1))]

#%%

def add_weight_noise(model, alpha = 0):
    #alpha defines how much the noise should be weighted compared to the original weights
    #alpha = 0 is no noise, 1 is all noise 
    for name, param in model.named_parameters():
        if name[:7] == "encoder" and name[-6:] == "weight":
            param.requires_grad = False
            noise = (alpha/100)*torch.randn(param.shape) #N(0,1) noise
            param *= (1-(alpha/100))
            param += noise
            param.requires_grad = True
            
    return model
            
#%%
#explore weight values
#all range between -2,2, std between 0.03,0.07

for name, param in model2.named_parameters():
    if name[:7] == "encoder" and name[-6:] == "weight":
        print("min:", np.min(param.detach().numpy()), "max:", np.max(param.detach().numpy()), "std:",np.std(param.detach().numpy()))
        #print(param.detach().numpy())

#and randn values

print("min:", np.min(noise.detach().numpy()), "max:", np.max(noise.detach().numpy()), "std:",np.std(noise.detach().numpy()))
   
#%%
m1 = BertModel.from_pretrained("prajjwal1/bert-small")
model = BertModel.from_pretrained("prajjwal1/bert-small")
m2 = perturb_weights(model = "prajjwal1/bert-small", alpha = 100)
m2 = add_weight_noise(m2, alpha = 100)
