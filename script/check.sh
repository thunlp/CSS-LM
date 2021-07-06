#!/bin/bash

FINAL_file=final_n_result_open_domain
RETRIVER_file=retriver_n_result_open_domain
BASELINE_file=baseline_n_result_open_domain
mkdir $FINAL_file
mkdir $RETRIVER_file
mkdir $BASELINE_file
rm -rf $FINAL_file/*
rm -rf $RETRIVER_file/*
rm -rf $BASELINE_file/*

#BATCH_SIZE=16
RETRIVE_BATCH=3072
BATCH_SIZE=3
NUM_LABEL=3
TRAIN_GPU=4,5,6
EVAL_GPU=4,5,6

#1, 10 --> 62...

#8 16 24
#for N in 1 5 10 20 30 40 50
#for N in 10 20 30 40 50
#for N in 5 10 15 20 30
for N in 5 10 15 20
do

    N_times_1=30
    N_times_2=30
    LEARNING_RATE=2e-5

    #N_times_1=1
    #N_times_2=1

    #if (( $N < $BATCH_SIZE ))
    #then
    #    N_times_1=1
    #    N_times_2=1
    #fi

    mkdir data/restaurant_fewshot
    cd data/restaurant_fewshot
    python3 extract_instance.py train_all.json $N
    cd ../..

    ###########
    ###Train Baseline models
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain"
    rm -rf $OUTFILE
    mkdir $OUTFILE

    MODEL=roberta-base
    DATA_in=data/restaurant_fewshot/

    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/finetune_roberta_sentiment_class_noaspect.py   --num_labels_task 3 --do_train   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16


    ###########
    #####Eval Baseline models
    ###########
    #CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/eval_roberta_sentiment_class_noaspect.py   --num_labels_task 3 --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/eval_roberta_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep.py   --num_labels_task 3 --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16


    python3 code/score.py $OUTFILE


    ###########
    #####Recorde the best retriver: Baseline
    ###########
    python3 code/extract.py $OUTFILE $BASELINE_file $N test

done
