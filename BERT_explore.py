#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:15:23 2023

@author: anna
"""

from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("prajjwal1/bert-small") 

#access attention layer
#can access query, key, value weights
model.encoder.layer[0].attention.self.query.weight

#access intermediate layer -- weight holds 4096x 1024 length tensors
#a dense tensor layer, with "hidden states"
model.encoder.layer[0].intermediate.dense.weight

#access output layer -- weight holds 1024x 4096 length tensors
#dense layer with layerNorm and dropout
model.encoder.layer[0].output.dense.weight.shape

#%%
#play with protBert

pbert = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
#pbert_inp = pbert.get_input_embeddings()
#pb_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
pbert.encoder.layer[0].output.dense.weight.shape

#change model to use bert embeddings
bert_inp = model.get_input_embeddings()
pbert.set_input_embeddings(bert_inp)

#set vocab size to match bert_tokenizer/embeddings
pbert.config.vocab_size = 30522

inp = tokenizer("This is a sentence in English.", return_tensors = "pt")

outp = model(**inp)

#sequence_Example = "A E T C Z A O"
#encoded_input = tokenizer(sequence_Example, return_tensors='pt')

#%%
#play with GLUE
#using NeMo scripts/tutorial....something is broken in the nemo package
# from nemo.collections import nlp as nemo_nlp
# from nemo.utils.exp_manager import exp_manager

# import os
# import wget 
# import torch
# import pytorch_lightning as pl
# from omegaconf import OmegaConf

# WORK_DIR = "work_dir"
# tasks = ["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI", "QNLI", "RTE", "WNLI"]
# MODEL_CONFIG = 'glue_benchmark_config.yaml'
# BRANCH = 'r1.15.0'

# #download model config file
# config_dir = WORK_DIR + '/configs/'
# os.makedirs(config_dir, exist_ok=True)
# if not os.path.exists(config_dir + MODEL_CONFIG):
#     print('Downloading config file...')
#     wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/nlp/glue_benchmark/' + MODEL_CONFIG, config_dir)
# else:
#     print ('config file is already exists')

    
# for t in tasks:
#     config_path = f'{WORK_DIR}/configs/{MODEL_CONFIG}'
#     print(config_path)
#     config = OmegaConf.load(config_path)
#     print(OmegaConf.to_yaml(config))
    
#     DATA_DIR = "glue_data/"+t
#     config.model.task_name = t
#     config.model.output_dir = WORK_DIR
#     config.model.dataset.data_dir = DATA_DIR
    
#     accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
#     config.trainer.devices = 1
#     config.trainer.accelerator = accelerator
    
#     config.trainer.precision = 16 if torch.cuda.is_available() else 32
    
#     # for mixed precision training, uncomment the line below (precision should be set to 16 and amp_level to O1):
#     # config.trainer.amp_level = O1
    
#     # remove distributed training flags
#     config.trainer.strategy = None
    
#     # setup max number of steps to reduce training time for demonstration purposes of this tutorial
#     config.trainer.max_steps = 128
    
#     trainer = pl.Trainer(**config.trainer)
#%%
#play with GLUE

#save model with .save_pretrained()
#should also save config file
#save tokenizer? Or can I specify the existing tokenizer even if it's not the same as the model name?

pbert.save_pretrained("protbert_engEmbeddings")
tokenizer.save_pretrained("bert_large_uncased")


#%%
#notes for cmd line on gpu06 server
source py_39/bin/activate
nohup sh glue_bash.sh > glue.out 2>&1 &
ps aux | grep glue
kill -9 <pid>

#%%
#freeze layers and save to use in glue
for layer in model.encoder.layer:
    layer.requires_grad = False

pbert_eng = BertForSequenceClassification.from_pretrained("protbert_engEmbeddings")    
model.save_pretrained("bert_frozen_all")

for name, param in newMod.named_parameters():
     print(name, param.requires_grad)

for name, param in model.named_parameters():
     param.requires_grad = False
        
for param in model.encoder.layer[0].parameters():
    param.requires_grad = True
    
#%%

config = BertConfig()
newMod = BertModel(config)
bert_inp = model.get_input_embeddings()
newMod.set_input_embeddings(bert_inp)

#%%
#add ff layers to end of model

import torch
import torch.nn as nn
from transformers import AutoModel, BertPreTrainedModel, BertConfig, BertForSequenceClassification, AutoModelForSequenceClassification

class ExtendedBERT (BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(self.config.model_name_or_path)
        self.classifier = nn.ModuleDict(
            {f"fc{i}":nn.Linear(self.config.hidden_size, self.config.intermediate_size) for i in range(self.config.num_fc)})
        self.post_init()


config = BertConfig.from_pretrained("bert-base-uncased")
config.model_name_or_path = "bert-base-uncased"
config.num_fc = 2
model = ExtendedBERT(config)

# model = ExtendedBERT(config, 2)

# def build_bert(model_name_or_path, model_config, num_fc):
# 	config = BertConfig.from_pretrained(model_config)
# 	return ExtendedBERT(model_name_or_path, config, num_fc)


# model = build_bert("bert-base-uncased", "bert-base-uncased", 3)
model.save_pretrained("bert_2layer_manualsave")
mod_load = ExtendedBERT.from_pretrained("bert_2layer_manualsave")
#%%
model = ExtendedBERT.from_pretrained("bert-base-uncased_2layers")
pbert = BertModel.from_pretrained("protbert_engEmbeddings")

model2 = BertForSequenceClassification.from_pretrained("bert-base-uncased")

#%%
    class additional_classifiers (nn.Module):
        def __init__(self, config, bert_pooler, num_fc):
            super().__init__()
            self.config = config
            self.num_fc = num_fc
            self.bert_pooler = bert_pooler
            
            self.fc = nn.ModuleDict()
            for i in range(self.num_fc):
                self.fc[f"dropout{i}"] = nn.Dropout(self.config.hidden_dropout_prob)
                self.fc[f"fc{i}"] = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            
        def forward(self, *x):
            
            out = self.bert_pooler(*x)
            for i in range(self.num_fc):
                out = self.fc[f"dropout{i}"](out)
                out = self.fc[f"fc{i}"](out)
            return (out)


model2.bert.pooler = additional_classifiers(model2.config, model2.bert.pooler, 2)


model2.bert.encoder.
model2.config

#%%
from transformers import EsmModel

esm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
bert_inp = model.get_input_embeddings()
esm.set_input_embeddings(bert_inp)

#set vocab size to match bert_tokenizer/embeddings
esm.config.vocab_size = 30522

esm.encoder.layer[0].output.dense.weight.shape

#%%
from datasets import load_dataset

data_args = DataTrainingArguments(task_name = "mrpc", max_seq_length = 128)
training_args = TrainingArguments(output_dir = "output_folder", do_train = True, do_eval = True, per_gpu_train_batch_size = 4,
                                  learning_rate = 2e-5, num_train_epochs = 3.0, gradient_accumulation_steps = 8, overwrite_output_dir = True)
model_args = ModelArguments("bert-base-uncased", ignore_mismatched_sizes = True)
# parser = HfArgumentParser((modeling_args, data, train))
# model_args, data_args, train_args = parser.parse_args_into_dataclasses()
raw_datasets = load_dataset(
    "glue",
    data_args.task_name,
    cache_dir=model_args.cache_dir,
    use_auth_token=True if model_args.use_auth_token else None,
)

label_list = raw_datasets["train"].features["label"].names
sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
num_labels = len(label_list)
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = {id: label for label, id in config.label2id.items()}

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]

metric = evaluate.load("glue", "mrpc")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()