#!/bin/bash

FINAL_file=final_bert_n_result_open_domain_entropy_sscl
RETRIVER_file=retriver_bert_n_result_open_domain_entropy_sscl
BASELINE_file=baseline_bert_n_result_open_domain_entropy_sscl
###
mkdir $FINAL_file
mkdir $RETRIVER_file
mkdir $BASELINE_file
rm -rf $FINAL_file/*
rm -rf $RETRIVER_file/*
rm -rf $BASELINE_file/*


RETRIVE_BATCH=1024

BATCH_SIZE=${10}
BATCH_SIZE_EVAL=180
NUM_LABEL=3

TRAIN_GPU=$1,$2,$3,$4
EVAL_GPU=$1,$2,$3,$4

INSTANCE_1=40000
INSTANCE_2=10000



ITER=${12}
MAX_LENGTH=${11}
for N in $5

do
    N_times_1=$8
    N_times_2=$9
    '''
    N_times_1=3
    N_times_2=3
    '''
    LEARNING_RATE=2e-5
    #LEARNING_RATE=4e-5

    mkdir ../data/restaurant_fewshot
    cd ../data/restaurant_fewshot
    cp train.json_$N train.json
    cd ../../script


    ###########
    ###Train Baseline models
    ###########
    OUTFILE="output_pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl"
    rm -rf $OUTFILE
    mkdir $OUTFILE

    #MODEL=roberta-base
    MODEL=bert-base-uncased
    ###
    DATA_in=../data/restaurant_fewshot/
    DATA_out=../data/opendomain_finetune_noword_10000/


    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3 -W ignore::UserWarning ../code/init_bert_sscl.py  --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length $MAX_LENGTH --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --augment_times 20 --K 16


    OUTFILE="output_pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl"
    #MODEL=roberta-base
    MODEL=bert-base-uncased
    ###
    DATA_in=../data/restaurant_fewshot/

    ###########
    #####Eval Baseline models
    ###########
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3 -W ignore::UserWarning ../code/eval_bert_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch_self.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length $MAX_LENGTH --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --choose_eval_test_both 2

    python3 ../code/score.py $OUTFILE



    ###########
    #####Recorde the best retriver: Baseline
    ###########
    mkdir $BASELINE_file"_"${ITER}
    rm $BASELINE_file"_"${ITER}/*
    python3 ../code/extract_test_by_eval.py $OUTFILE $BASELINE_file"_"${ITER} $N



done
