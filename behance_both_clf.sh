#!/bin/bash

OUTPUT_ROOT=../models/bond/clf
SCRIPT=./bond/run_classification.py

if [ "$1" = "--multimodal" ]; then
  echo "Using multi-modal model"
  SCRIPT=./multimodal_bond/run_classification.py
  OUTPUT_ROOT=../models/multimodal_bond/clf
fi

GPUID=$2
echo "Run on GPU $GPUID"

# data
# PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
PROJECT_ROOT=.
DATA_ROOT=$PROJECT_ROOT/../data/intent/both

# model
MODEL_TYPE=roberta
MODEL_NAME=roberta-base

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=200

TRAIN_BATCH=32
EVAL_BATCH=64

# self-training parameters
REINIT=0
BEGIN_STEP=100000000 #high BEGIN_STEP prevents self training
LABEL_MODE=soft
PERIOD=450
HP_LABEL=5.9

EPOCH=5
SEED=0
WEIGHT_DECAY=1e-4

# params
LR=1e-5

for LR in 1e-4 5e-5 1e-5 
do
	for TRAIN_BATCH in 32 16
	do
		# output
		OUTPUT=$OUTPUT_ROOT/baseline/${MODEL_TYPE}_${TRAIN_BATCH}_${LR}/

		[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
		cp -f $(readlink -f "$0") $OUTPUT/script
		rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'dataset' --exclude 'pretrained_model' --exclude 'outputs' $PROJECT_ROOT/ $OUTPUT/src

		CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python -m $SCRIPT --data_dir $DATA_ROOT \
		  --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME \
		  --learning_rate $LR \
		  --weight_decay $WEIGHT_DECAY \
		  --adam_epsilon $ADAM_EPS \
		  --adam_beta1 $ADAM_BETA1 \
		  --adam_beta2 $ADAM_BETA2 \
		  --num_train_epochs $EPOCH \
		  --warmup_steps $WARMUP \
		  --per_gpu_train_batch_size $TRAIN_BATCH \
		  --per_gpu_eval_batch_size $EVAL_BATCH \
		  --logging_steps 200 \
		  --save_steps 1500 \
		  --do_train \
		  --do_eval \
		  --do_predict \
		  --evaluate_during_training \
		  --output_dir $OUTPUT \
		  --cache_dir $PROJECT_ROOT/pretrained_model \
		  --seed $SEED \
		  --max_seq_length 128 \
		  --overwrite_output_dir \
		  --eval_all_checkpoints
		  
		break
	done
	break
done
