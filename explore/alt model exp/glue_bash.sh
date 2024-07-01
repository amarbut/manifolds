#!/bin/bash

#bert_base_untrained
#pbert_engEmbeddings
#bert-base-uncased

#TASK_NAME=MRPC
MODEL="prajjwal1/bert-small"
CONFIG="prajjwal1/bert-small"
TOKENIZER="prajjwal1/bert-small"

#for TASK_NAME in cola sst2 mrpc stsb qqp mnli qnli rte wnli
#for TASK_NAME in cola sst2 qqp mnli qnli rte wnli
for TASK_NAME in cola
do
python3 run_glue.py \
  --model_name_or_path $MODEL \
  --overwrite_output_dir \
  --config_name $CONFIG \
  --tokenizer_name $TOKENIZER \
  --ignore_mismatched_sizes True \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --logging_steps 50 \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /media/anna/Samsung_T5/manifolds/glue_results/$MODEL/$TASK_NAME/ \
  --gradient_accumulation_steps 8 #\
#  --frozen_layers -1 #\
#  --num_fc 4

done
