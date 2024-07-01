#!/bin/bash

SAVE_LOC=alt_model_save/
PAPER=

for MODEL_NAME in 
do
	nohup python3 -u sample_build_and_metrics.py \
	--model $MODEL_NAME \
	--paper $PAPER \
	--save_loc $SAVE_LOC \
	>> alt_model_run.log 2>&1
done
