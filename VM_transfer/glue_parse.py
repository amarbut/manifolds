#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:27:49 2024

@author: anna
"""

import os
import json
import numpy as np
import argparse

glue_dict = {'cola': "eval_matthews_correlation",
             'mnli': "eval_accuracy",
             'mrpc': "eval_combined_score",
             'qnli': "eval_accuracy",
             'qqp': "eval_combined_score",
             'rte': "eval_accuracy",
             'sst2': "eval_accuracy",
             'stsb': "eval_combined_score",
             'wnli': "eval_accuracy"}

def glue_parse(folder_loc):
    
    results = dict()
    for task in os.listdir(folder_loc):
        if task != "all_results.json":
            r_name = glue_dict[task]
            task_dir = folder_loc+"/"+task
            for filename in os.listdir(task_dir):
                if filename == "all_results.json":
                    file_loc = task_dir+"/"+filename
                    with open(file_loc, "r") as f:
                        r = json.load(f)
                        results[task] = r[r_name]
    avg = np.mean(list(results.values()))
    results['average'] = avg
    
    with open(folder_loc+"/all_results.json", "w") as save_file:
        json.dump(results, save_file)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_loc', help = 'Folder holding glue task folders', default = '', required = False)
    args = vars(parser.parse_args())
    
    glue_parse(**args)
    