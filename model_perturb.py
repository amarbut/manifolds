#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:04:48 2024

@author: anna
"""

from transformers import BertModel, RobertaForMaskedLM
import torch
import argparse
import os

seed_dict = {"1":111,
             "2":222,
             "3":333}

def perturb_weights(model, alpha, model_type):
    
    if model_type == "bert":
        m = BertModel.from_pretrained(model)
        #alpha = 0 is no noise, 100 is all noise 
        for name, param in m.named_parameters():
            if name[:7] == "encoder" and name[-6:] == "weight":
                param.requires_grad = False
                noise = (alpha/100)*torch.randn(param.shape) #N(0,1) noise
                param *= (1-(alpha/100))
                param += noise
                param.requires_grad = True
    elif model_type == "roberta":
        m = RobertaForMaskedLM.from_pretrained(model)
        #alpha = 0 is no noise, 100 is all noise 
        for name, param in m.named_parameters():
            if name[8:15] == "encoder" and name[-6:] == "weight":
                param.requires_grad = False
                noise = (alpha/100)*torch.randn(param.shape) #N(0,1) noise
                param *= (1-(alpha/100))
                param += noise
                param.requires_grad = True
            
    return m

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help = 'Name of pre-trained Bert model', default = 'prajjwal1/bert-small', required = False)
    parser.add_argument('--alpha', help = 'How much the noise should be weighted compared to the original weights', type = float, default = 0, required = False)
    parser.add_argument('--save_loc', help = 'Location for saving perturbed model', default = '', required = False)
    parser.add_argument('--run', help = 'Number run with this model for filename', default = '1', required = False)
    parser.add_argument('--model_type', help = 'bert or roberta', default = 'bert', required = False)
    
    
    args = vars(parser.parse_args())
    set_seed = seed_dict[str(args["run"])]
    
    m = perturb_weights(args["model"], args["alpha"], args['model_type'])
    
    if args["alpha"] % 1 == 0:
        alpha = str(int(args["alpha"]))
    else:
        alpha = str(args["alpha"])
    file_name = args["save_loc"]+args["model"]+ "_" + alpha + "_" + str(args["run"])
    print(file_name)
    os.makedirs(args["save_loc"], exist_ok = True)
    m.save_pretrained(file_name)
    
