#!/bin/bash

FINAL_file=ablation_both
mkdir $FINAL_file
rm -rf $FINAL_file/*

#BATCH_SIZE=16
RETRIVE_BATCH=10
BATCH_SIZE=16
NUM_LABEL=3
TRAIN_GPU=0,7
EVAL_GPU=0,7
#INSTANCE=10000
#INSTANCE=100
INSTANCE=16
MODE="both"

#1, 10 --> 62...

for N in 10
do
    ###
    '''
    N_times_1=30
    N_times_2=30
    '''
    N_times_1=1
    N_times_2=1
    ###
    LEARNING_RATE=2e-5


    #mkdir data/restaurant_fewshot
    #cd data/restaurant_fewshot
    #python3 extract_instance.py train_all.json $N
    #cd ../..


    ###########
    #####Retrive 100 data
    ###########
    INPUT=data/opendomain_fewshot/
    OUTPUT=../ablation/
    MODEL=roberta-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain
    DATA_in=data/restaurant_fewshot/
    mkdir OUTPUT
    rm -rf OUTPUT/*

    #--output_dir $OUTPUT
    #data_dir_indomain $DATA_in
    #data_dir_outdomain $INPUT
    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/ablation_both.py --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $INPUT  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $RETRIVE_BATCH --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_2   --output_dir $OUTPUT  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K $INSTANCE --mode $MODE


done
