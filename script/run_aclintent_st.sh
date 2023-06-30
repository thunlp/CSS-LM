#!/bin/bash

FINAL_file=final_n_result_open_domain_aclintent_entropy_st
RETRIVER_file=retriver_n_result_open_domain_aclintent_entropy_st
BASELINE_file=baseline_n_result_open_domain_aclintent_entropy_st
mkdir $FINAL_file
mkdir $RETRIVER_file
mkdir $BASELINE_file
rm -rf $FINAL_file/*
rm -rf $RETRIVER_file/*
rm -rf $BASELINE_file/*

RETRIVE_BATCH=1024
BATCH_SIZE=${10}
BATCH_SIZE_EVAL=180
NUM_LABEL=6

TRAIN_GPU=$1,$2,$3,$4
EVAL_GPU=$1,$2,$3,$4

INSTANCE_1=40000
INSTANCE_2=10000


ITER=${12}
for N in $5
do

    N_times_1=$8
    N_times_2=$9

    LEARNING_RATE=2e-5


    mkdir ../data/aclintent_fewshot
    cd ../data/aclintent_fewshot
    cp train.json_$N train.json
    cd ../../script

    ###########
    ###Train Baseline models
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_aclintent_entropy_st"


    ######################################################
    ######################################################
    ####################My model##########################
    ######################################################
    ######################################################

    for K in 16 32 48
    do
        ###########
        #####Train a model and better retriver for generation doc representation
        ###########
        OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_aclintent_entropy_st"
        rm -rf $OUTFILE
        mkdir $OUTFILE
        cp -r backup_aclintent/* $OUTFILE/
        INPUT=../data/openwebtext/train.txt
        OUTPUT=../data/opendomain_finetune_aclintent_10000
        MODEL=roberta-base-768-aclintent-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
        mkdir $OUTPUT
        mkdir $MODEL
        rm -rf $MODEL/*
        cp -r roberta-base-768/* $MODEL/
        rm -rf $MODEL/pytorch_model.bin
        cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin

        rm -rf $OUTFILE
        mkdir $OUTFILE

        ##From org train
        #MODEL=roberta-base
        #From trained <s> train
        MODEL=roberta-base-768-aclintent-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self

        #Need to filter some sentence to train the domain model
        #Given and put some outdomain data in DATA_out
        DATA_out=../data/opendomain_finetune_aclintent_10000/
        DATA_in=../data/aclintent_fewshot/

        CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3 -W ignore::UserWarning ../code/self_training.py  --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --augment_times 20 --K $K



        ###########
        #####Eval the model and extract the best model to _gen dir
        ###########

        MODEL=roberta-base-768-aclintent-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
        DATA_in=../data/aclintent_fewshot/
        DATA_out=../data/opendomain_finetune_aclintent_10000/
        OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_aclintent_entropy_st"


        #eval
        #eval  --> can only be on one GPU
        ###Extract and eval  model
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python3 -W ignore::UserWarning ../code/eval_roberta_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch_self.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --choose_eval_test_both 2

        python3 ../code/score.py $OUTFILE


        ###########
        #####Recorde the best retriver: Baseline
        ###########
        RETRIVER_file=retriver_n_result_open_domain_aclintent_entropy_st_$K"_"$ITER
        mkdir $RETRIVER_file
        rm -rf $RETRIVER_file/*

        python3 ../code/extract_test_by_eval.py $OUTFILE $RETRIVER_file $N

    done

done
