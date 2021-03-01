#!/bin/bash

#FINAL_file=final_n_result_open_domain
#RETRIVER_file=retriver_n_result_open_domain
#BASELINE_file=baseline_n_result_open_domain
mkdir $FINAL_file
mkdir $RETRIVER_file
mkdir $BASELINE_file
#rm -rf $FINAL_file/*
#rm -rf $RETRIVER_file/*
#rm -rf $BASELINE_file/*

BATCH_SIZE=16
RETRIVE_BATCH=16
BATCH_SIZE=4
NUM_LABEL=3
TRAIN_GPU=4,5,6
EVAL_GPU=4,5,6
###
INSTANCE_1=20000
INSTANCE_2=20000
#INSTANCE=100
###

#1, 10 --> 62...

#8 16 24
#for N in 1 5 10 20 30 40 50
#for N in 10 20 30 40 50
#for N in 5 10 15 20 30
###
#for N in 5 10 15 20
for N in 5
###
do
    ###
    '''
    N_times_1=30
    N_times_2=30
    '''
    N_times_1=3
    N_times_2=3
    ###
    LEARNING_RATE=2e-5


    '''
    mkdir data/restaurant_fewshot
    cd data/restaurant_fewshot
    python3 extract_instance.py train_all.json $N
    cd ../..
    '''

    ###########
    ###Train Baseline models
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain"
    #INPUT=data/openwebtext/train.txt
    INPUT=data/openwebtext/train_100.txt
    DATA_in=data/restaurant_fewshot/
    OUTPUT=data/opendomain_finetune_noword_10000/
    MODEL=roberta-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain

    #--output_dir $OUTPUT
    #data_dir_indomain $DATA_in
    #data_dir_outdomain $INPUT
    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/gen_test.py --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $INPUT  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $RETRIVE_BATCH --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_2   --output_dir $OUTPUT  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K $INSTANCE_2


done
