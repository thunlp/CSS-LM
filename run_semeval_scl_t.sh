#!/bin/bash

FINAL_file=final_n_result_open_domain_entropy_scl_t
RETRIVER_file=retriver_n_result_open_domain_entropy_scl_t
BASELINE_file=baseline_n_result_open_domain_entropy_scl_t
mkdir $FINAL_file
mkdir $RETRIVER_file
mkdir $BASELINE_file
rm -rf $FINAL_file/*
rm -rf $RETRIVER_file/*
rm -rf $BASELINE_file/*

#BATCH_SIZE=16
RETRIVE_BATCH=3072
#BATCH_SIZE=8
#BATCH_SIZE=4
BATCH_SIZE=${10}
BATCH_SIZE_EVAL=180
NUM_LABEL=3
#TRAIN_GPU=0,1,2,3
#TRAIN_GPU=4,5,6,7
#EVAL_GPU=4,5,6,7
#TRAIN_GPU=1,2,3,4
#EVAL_GPU=1,2,3,4
#TRAIN_GPU=0,1,2,3
#EVAL_GPU=0,1,2,3
TRAIN_GPU=$1,$2,$3,$4
EVAL_GPU=$1,$2,$3,$4
#TRAIN_GPU=0,2,5,7
#EVAL_GPU=0,2,5,7
#TRAIN_GPU=1,2,5,7
#EVAL_GPU=1,2,5,7
###
INSTANCE_1=36603
INSTANCE_2=10000
#INSTANCE=100
###

#1, 10 --> 62...

#8 16 24
#for N in 1 5 10 20 30 40 50
#for N in 10 20 30 40 50
#for N in 5 10 15 20 30
###
#for N in 5 10 15 20
#for N in "all"
#for N in $6 $7 $8
for N in $5
#for N in 25 50 100
#for N in 5
#for N in "all"
###
do
    ###
    N_times_1=$8
    N_times_2=$9
    '''
    N_times_1=3
    N_times_2=3
    '''
    ###
    LEARNING_RATE=2e-5
    #LEARNING_RATE=4e-5

    ###
    mkdir data/restaurant_fewshot
    cd data/restaurant_fewshot
    #N==50 was change
    #python3 extract_instance.py train_all.json $N
    cp train.json_$N train.json
    #cp train_$N.json train.json
    cd ../..


    ###########
    ###Train Baseline models
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_scl_t"
    rm -rf $OUTFILE
    mkdir $OUTFILE

    MODEL=roberta-base
    DATA_in=data/restaurant_fewshot/
    DATA_out=data/opendomain_finetune_noword_10000/


    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/init_scl_t.py  --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K 16

    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_scl_t"
    MODEL=roberta-base
    DATA_in=data/restaurant_fewshot/

    ###########
    #####Eval Baseline models
    ###########
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/eval_roberta_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch_self.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1
    --task 2 --fp16 --choose_eval_test_both 2

    python3 code/score.py $OUTFILE



    ###########
    #####Recorde the best retriver: Baseline
    ###########
    python3 code/extract.py $OUTFILE $BASELINE_file $N test


done
