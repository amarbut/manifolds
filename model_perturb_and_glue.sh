#!/bin/bash

MODEL_NAME='prajjwal1/bert-small'
ALPHA=1
SAVE_LOC=/media/anna/Samsung_T5/manifolds/

python3 model_perturb.py \
	--model $MODEL_NAME \
	--alpha $ALPHA \
	--save_loc $SAVE_LOC


MODEL=${SAVE_LOC}${MODEL_NAME}_${ALPHA}

for TASK_NAME in cola #sst2 mrpc stsb qqp mnli qnli rte wnli
do
python3 run_glue.py \
  --model_name_or_path $MODEL \
  --overwrite_output_dir \
  --config_name $MODEL_NAME \
  --tokenizer_name $MODEL_NAME \
  --ignore_mismatched_sizes True \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --logging_steps 50 \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /${SAVE_LOC}/glue_results/${MODEL_NAME}_${ALPHA}/$TASK_NAME/ \
  --gradient_accumulation_steps 8 

done