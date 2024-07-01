#!/bin/bash

MODEL_NAME='bert-base-uncased'
#ALPHA=1
SAVE_LOC=/media/anna/Samsung_T5/manifolds/base_compare/
SEQUENCE_LOC=sample_sequences.pkl


for ALPHA in 1 3 5
do
  for RUN in 1
  do
    echo "perturbing model alpha ${ALPHA} run ${RUN}" 
    nohup python3 -u model_perturb.py \
  	--model $MODEL_NAME \
 	--alpha $ALPHA \
	--save_loc $SAVE_LOC \
        --run $RUN \
        > build.log 2>&1

    SAVE_ALPHA=${ALPHA%.*}
    echo ${SAVE_ALPHA}
    MODEL=${SAVE_LOC}${MODEL_NAME}_${ALPHA}_${RUN}
    MODEL2=${MODEL_NAME}_${ALPHA}_${RUN}

    for TASK_NAME in cola sst2 mrpc stsb qqp mnli qnli rte wnli
    do
    echo "running ${TASK_NAME}"
    nohup python3 -u run_glue.py \
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
      --output_dir ${SAVE_LOC}glue_results/${MODEL2}/$TASK_NAME/ \
      --gradient_accumulation_steps 8 \
      > glue.log 2>&1

    done

    echo "parsing glue results"
    nohup python3 -u glue_parse.py \
      --folder_loc ${SAVE_LOC}glue_results/${MODEL2} \
      > parse.log 2>&1 

    echo "latent space metrics"
    nohup python3 -u sample_build_and_metrics.py \
      --model $MODEL \
      --sequences $SEQUENCE_LOC \
      --save_loc ${SAVE_LOC}metric_results/${MODEL2}\
      > metrics.log 2>&1
  done
done


 
