#!/bin/bash

#OUTFILE=save_emb
#mkdir $BASELINE_file
#rm -rf $BASELINE_file/*

#BATCH_SIZE=16
RETRIVE_BATCH=3072
#BATCH_SIZE=8
#BATCH_SIZE=4
#BATCH_SIZE=4
BATCH_SIZE_EVAL=180
#BATCH_SIZE_EVAL=1
#NUM_LABEL=3
NUM_LABEL=3
#TRAIN_GPU=$1,$2,$3,$4
#EVAL_GPU=$1,$2,$3,$4
EVAL_GPU=7
###
INSTANCE_1=36603
INSTANCE_2=10000
#INSTANCE=100
###

#1, 10 --> 62...

#for N in $5
#sscl_dt --> use 32
for N in 16
do
    ###
    #N_times_1=$8
    #N_times_2=$9
    N_times_1=1
    N_times_2=1
    ###
    LEARNING_RATE=2e-5
    #LEARNING_RATE=4e-5

    ###
    #mkdir data/restaurant_fewshot
    cd data/semeval_abs
    #N==50 was change
    #python3 extract_instance.py train_all.json $N
    python3 extract_instance.py
    cp res_lap_comb.json test.json
    #cp train_$N.json train.json
    cd ../..

    #change OUTFILE
    #OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_finetune_all"
    #OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_finetune"
    #OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl"
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_dt"
    #OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_scl_dt"
    MODEL=roberta-base
    DATA_in=data/semeval_abs/

    ###########
    #####Eval Baseline models
    ###########
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/draw.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1  --task 2 --fp16 --choose_eval_test_both 1

    #python3 code/score.py $OUTFILE



    ###########
    #####Recorde the best retriver: Baseline
    ###########
    #python3 code/extract.py $OUTFILE $BASELINE_file $N test


done
