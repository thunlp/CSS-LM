#!/bin/bash

FINAL_file=final_n_result_open_domain_entropy_sscl_t
RETRIVER_file=retriver_n_result_open_domain_entropy_sscl_t
BASELINE_file=baseline_n_result_open_domain_entropy_sscl_t
mkdir $FINAL_file
mkdir $RETRIVER_file
mkdir $BASELINE_file
rm -rf $FINAL_file/*
rm -rf $RETRIVER_file/*
rm -rf $BASELINE_file/*

#BATCH_SIZE=16
RETRIVE_BATCH=1024
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
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_t"
    rm -rf $OUTFILE
    mkdir $OUTFILE

    MODEL=roberta-base
    DATA_in=data/restaurant_fewshot/
    DATA_out=data/opendomain_finetune_noword_10000/


    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/init_sscl_t.py  --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K 16

    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_t"
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


    ##########
    ###Initial retrive (First epoch retrive): Assume: pytorch_model.bin is retriver
    #####Retrive 10000 data
    #########
    ###
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_t"
    ###
    INPUT=data/openwebtext/train.txt
    #INPUT=data/openwebtext/train_100.txt
    OUTPUT=data/opendomain_finetune_noword_10000
    MODEL=roberta-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
    mkdir $OUTPUT
    mkdir $MODEL
    rm -rf $MODEL/*
    cp -r roberta-base-768/* $MODEL/
    rm -rf $MODEL/pytorch_model.bin
    #cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin
    cp $OUTFILE/pytorch_model.bin_test_best $MODEL/pytorch_model.bin

    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3 code/gen_yelp_dataset_roberta_task_finetune_HEADandTAIL_baseline_batch_self.py $INPUT $OUTPUT/train $MODEL $INSTANCE_1 $NUM_LABEL $RETRIVE_BATCH


    ######################################################
    ######################################################
    ####################My model##########################
    ######################################################
    ######################################################


    ###########
    #####Train a model and better retriver for generation doc representation
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_t"
    rm -rf $OUTFILE
    mkdir $OUTFILE

    ##From org train
    #MODEL=roberta-base
    #From trained <s> train
    MODEL=roberta-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self

    #Need to filter some sentence to train the domain model
    #Given and put some outdomain data in DATA_out
    DATA_out=data/opendomain_finetune_noword_10000/
    DATA_in=data/restaurant_fewshot/

    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_updateRep_opendomain_fast_entropy_self.py   --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K 16


    ###########
    #####Eval the model and extract the best model to _gen dir
    ###########

    MODEL=roberta-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
    DATA_in=data/restaurant_fewshot/
    DATA_out=data/opendomain_finetune_noword_10000/
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_t"
    #rm -rf $MODEL
    #cp -r roberta-base-768 $MODEL
    #rm $MODEL/pytorch_model.bin
    #cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin
    #OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL"

    #eval
    #eval  --> can only be on one GPU
    ###Extract and eval  model
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/eval_roberta_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch_self.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1  --task 2 --fp16 --choose_eval_test_both 2

    python3 code/score.py $OUTFILE




    ###########
    #####Recorde the best retriver
    ###########
    #python3 code/extract.py $OUTFILE $RETRIVER_file $N eval
    python3 code/extract.py $OUTFILE $RETRIVER_file $N test

    exit

    rm -rf $MODEL
    cp -r roberta-base-768 $MODEL
    rm $MODEL/pytorch_model.bin
    cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin


    ###########
    #####Retrive 10000 data
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_t"
    INPUT=data/openwebtext/train.txt
    #INPUT=data/openwebtext/train_100.txt
    OUTPUT=data/opendomain_finetune_noword_10000/
    MODEL=roberta-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
    mkdir $MODEL
    rm -rf $MODEL/*
    cp -r roberta-base-768/* $MODEL/
    rm -rf $MODEL/pytorch_model.bin
    cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin

    #CUDA_VISIBLE_DEVICES=$EVAL_GPU python3 code/gen_yelp_dataset_roberta_task_finetune_HEADandTAIL.py $INPUT $OUTPUT $MODEL $INSTANCE
    #--output_dir $OUTPUT
    #data_dir_indomain $DATA_in
    #data_dir_outdomain $INPUT
    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/gen_yelp_dataset_roberta_task_finetune_HEADandTAIL_updateRep_bottomANDtop_classifier_batch_AllLabel_self.py --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $INPUT  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $RETRIVE_BATCH --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_2   --output_dir $OUTPUT  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K $INSTANCE_2


    ###########
    #####Train a retriver for generation doc representation
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_t"
    rm $OUTFILE/*

    #MODEL=roberta-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL
    MODEL=roberta-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self

    DATA_out=data/opendomain_finetune_noword_10000/
    DATA_in=data/restaurant_fewshot/

    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_updateRep_opendomain_fast_entropy_self.py   --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_2   --output_dir $OUTFILE  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K 16


    ###########
    #####Eval test
    ###########
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_t"
    DATA_in=data/restaurant_fewshot/
    MODEL=roberta-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self

    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/eval_roberta_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch_self.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_2   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1
    --task 2 --fp16 --choose_eval_test_both 2

    python3 code/score.py $OUTFILE


    ###########
    #####Extract best test
    ###########
    #OUTFILE=output_finetune_roberta_yelp_DomainTask_useMLMCLASS_noaspect
    OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_t"
    python3 code/extract.py $OUTFILE $FINAL_file $N test


done
