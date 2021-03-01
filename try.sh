#!/bin/bash

FINAL_file=final_try
RETRIVER_file=retriver_try
BASELINE_file=baseline_try
mkdir $FINAL_file
mkdir $RETRIVER_file
mkdir $BASELINE_file
rm -rf $FINAL_file/*
rm -rf $RETRIVER_file/*
rm -rf $BASELINE_file/*

#BATCH_SIZE=16
RETRIVE_BATCH=3072
#BATCH_SIZE=8
BATCH_SIZE=16
BATCH_SIZE_EVAL=80
NUM_LABEL=6
TRAIN_GPU=0,1,2
EVAL_GPU=0,1,2

INSTANCE_1=31844
INSTANCE_2=10000
#INSTANCE=100

#1, 10 --> 62...

#8 16 24
#for N in 1 5 10 20 30 40 50
#for N in 10 20 30 40 50
#for N in 5 10 15 20
###
#for N in 5 10 15 20
for N in 5
do

    ###
    N_times_1=30
    N_times_2=30
    ###

    LEARNING_RATE=2e-5

    '''
    mkdir data/trec_fewshot
    cd data/trec_fewshot
    #python3 extract_instance.py train_all.json $N
    #cp train.json_$N train.json
    cp train_$N.json train.json
    cd ../..

    ###########
    ###Train Baseline models
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_trec"
    rm -rf $OUTFILE
    mkdir $OUTFILE

    MODEL=roberta-base
    DATA_in=data/trec_fewshot/

    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/finetune_roberta_sentiment_class_noaspect.py   --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16


    ###########
    #####Eval Baseline models
    ###########
    #CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/eval_roberta_sentiment_class_noaspect.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/eval_roberta_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16


    python3 code/score.py $OUTFILE


    ###########
    #####Recorde the best retriver: Baseline
    ###########
    python3 code/extract.py $OUTFILE $BASELINE_file $N test

    ##########
    ###Initial retrive (First epoch retrive): Assume: pytorch_model.bin is retriver
    #####Retrive 10000 data
    #########
    INPUT=data/openwebtext/train.txt
    OUTPUT=data/opendomain_finetune_trec_10000
    MODEL=roberta-base-768-trec-DomainTask-noword-sentiment-HEADandTAIL-opendomain
    mkdir $OUTPUT
    mkdir $MODEL
    rm -rf $MODEL/*
    cp -r roberta-base-768/* $MODEL/
    rm -rf $MODEL/pytorch_model.bin
    #cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin
    cp $OUTFILE/pytorch_model.bin_test_best $MODEL/pytorch_model.bin

    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3 code/gen_yelp_dataset_roberta_task_finetune_HEADandTAIL_baseline_batch.py $INPUT $OUTPUT/train $MODEL $INSTANCE_1 $NUM_LABEL $RETRIVE_BATCH


    ######################################################
    ######################################################
    ####################My model##########################
    ######################################################
    ######################################################


    ###########
    #####Train a model and better retriver for generation doc representation
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_trec"
    '''
    OUTFILE="try_del"
    rm -rf $OUTFILE
    mkdir $OUTFILE

    ##From org train
    #MODEL=roberta-base
    #From trained <s> train
    MODEL=roberta-base-768-trec-DomainTask-noword-sentiment-HEADandTAIL-opendomain

    #Need to filter some sentence to train the domain model
    #Given and put some outdomain data in DATA_out
    DATA_out=data/opendomain_finetune_trec_10000/
    DATA_in=data/trec_fewshot/
    #DATA_in=data/restaurant_fewshot/

    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_updateRep_opendomain_fast_crossentropy.py   --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K 8

    exit

    ###########
    #####Eval the model and extract the best model to _gen dir
    ###########

    MODEL=roberta-base-768-trec-DomainTask-noword-sentiment-HEADandTAIL-opendomain
    DATA_in=data/trec_fewshot/
    #rm -rf $MODEL
    #cp -r roberta-base-768 $MODEL
    #rm $MODEL/pytorch_model.bin
    #cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin
    #OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL"

    #eval
    #eval  --> can only be on one GPU
    ###Extract and eval  model
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/eval_roberta_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16

    python3 code/score.py $OUTFILE


    ###########
    #####Recorde the best retriver
    ###########
    #python3 code/extract.py $OUTFILE $RETRIVER_file $N eval
    python3 code/extract.py $OUTFILE $RETRIVER_file $N test

    rm -rf $MODEL
    cp -r roberta-base-768 $MODEL
    rm $MODEL/pytorch_model.bin
    cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin


    ###########
    #####Retrive 10000 data
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_trec"
    INPUT=data/openwebtext/train.txt
    OUTPUT=data/opendomain_finetune_trec_10000/
    MODEL=roberta-base-768-trec-DomainTask-noword-sentiment-HEADandTAIL-opendomain
    mkdir $MODEL
    rm -rf $MODEL/*
    cp -r roberta-base-768/* $MODEL/
    rm -rf $MODEL/pytorch_model.bin
    cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin

    #CUDA_VISIBLE_DEVICES=$EVAL_GPU python3 code/gen_yelp_dataset_roberta_task_finetune_HEADandTAIL.py $INPUT $OUTPUT $MODEL $INSTANCE
    #--output_dir $OUTPUT
    #data_dir_indomain $DATA_in
    #data_dir_outdomain $INPUT
    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/gen_yelp_dataset_roberta_task_finetune_HEADandTAIL_updateRep_bottomANDtop_classifier_batch_AllLabel.py --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $INPUT  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $RETRIVE_BATCH --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_2   --output_dir $OUTPUT  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K $INSTANCE_2


    ###########
    #####Train a retriver for generation doc representation
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_trec"
    rm $OUTFILE/*

    #MODEL=roberta-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL
    MODEL=roberta-base-768-trec-DomainTask-noword-sentiment-HEADandTAIL-opendomain

    DATA_out=data/opendomain_finetune_trec_10000/
    DATA_in=data/trec_fewshot/

    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_updateRep_opendomain.py   --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_2   --output_dir $OUTFILE  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K 16


    ###########
    #####Eval test
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_trec"
    DATA_in=data/trec_fewshot/
    MODEL=roberta-base-768-trec-DomainTask-noword-sentiment-HEADandTAIL-opendomain

    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/eval_roberta_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_2   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16

    python3 code/score.py $OUTFILE


    ###########
    #####Extract best test
    ###########
    #OUTFILE=output_finetune_roberta_yelp_DomainTask_useMLMCLASS_noaspect
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_trec"
    python3 code/extract.py $OUTFILE $FINAL_file $N test


done
