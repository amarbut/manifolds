#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:38:25 2024

@author: anna
"""

#glue_folder = "/media/anna/Samsung_T5/manifolds/base_compare/glue_results"
# for VM in os.listdir("/media/anna/Samsung_T5/manifolds/VM_results"):
#     glue_folder = "/media/anna/Samsung_T5/manifolds/VM_results/"+VM+"/test/glue_results/prajjwal1"
    
glue_folder = "/media/anna/Samsung_T5/manifolds/untrained_w_emb/glue_results"    
glue_parse(glue_folder)

for m in os.listdir(glue_folder):
    glue_parse(glue_folder+"/"+m)
    
    
alt_metrics["glue_average"] = [0.684, 0.703, 0.681,0.557, 0.767, 0.798, 0.752, 0.564, 0.841,0.623,0.776, 0.78, 0.557]
    