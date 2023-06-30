#!/bin/bash

FINAL_file=final_bert_n_result_open_domain_entropy_sscl_dt
RETRIVER_file=retriver_bert_n_result_open_domain_entropy_sscl_dt
BASELINE_file=baseline_bert_n_result_open_domain_entropy_sscl_dt
###
mkdir $FINAL_file
mkdir $RETRIVER_file
mkdir $BASELINE_file
rm -rf $FINAL_file/*
rm -rf $RETRIVER_file/*
rm -rf $BASELINE_file/*

#BATCH_SIZE=16
RETRIVE_BATCH=1024
BATCH_SIZE=${10}
BATCH_SIZE_EVAL=180
NUM_LABEL=3
TRAIN_GPU=$1,$2,$3,$4
EVAL_GPU=$1,$2,$3,$4

INSTANCE_1=40000
INSTANCE_2=10000

MAX_LENGTH=${11}
ITER=${12}
for N in $5

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
    mkdir ../data/restaurant_fewshot
    cd ../data/restaurant_fewshot
    cp train.json_$N train.json
    cd ../../script

    ###########
    ###Train Baseline models
    ###########

    OUTFILE="output_pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_dt"
    rm -rf $OUTFILE
    mkdir $OUTFILE


    #MODEL=roberta-base
    MODEL=bert-base-uncased
    ###
    DATA_in=../data/restaurant_fewshot/
    DATA_out=../data/opendomain_finetune_noword_10000/
    rm -rf $DATA_out
    cp -r ../download/opendomain_finetune_noword_10000 $DATA_out


    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3 -W ignore::UserWarning ../code/init_bert_sscl_dt.py  --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length $MAX_LENGTH --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --augment_times 20 --K 16


    #MODEL=roberta-base
    MODEL=bert-base-uncased
    ###
    DATA_in=../data/restaurant_fewshot/

    ###########
    #####Eval Baseline models
    ###########

    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3 -W ignore::UserWarning ../code/eval_bert_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch_self.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length $MAX_LENGTH --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --choose_eval_test_both 2
    ###

    python3 ../code/score.py $OUTFILE




    ###########
    #####Recorde the best retriver: Baseline
    ###########
    echo $BASELINE_file"_"${ITER}
    mkdir $BASELINE_file"_"${ITER}
    rm $BASELINE_file"_"${ITER}/*
    python3 ../code/extract_test_by_eval.py $OUTFILE $BASELINE_file"_"${ITER} $N



    ##########
    ###Initial retrive (First epoch retrive): Assume: pytorch_model.bin is retriver
    #####Retrive 10000 data
    #########
    ###
    ###
    INPUT=../data/openwebtext/train.txt
    OUTPUT=../data/opendomain_finetune_noword_10000
    #fix
    ###
    MODEL=bert-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
    ###
    mkdir $OUTPUT
    mkdir $MODEL
    rm -rf $MODEL/*
    cp -r bert-base-768/* $MODEL/
    rm -rf $MODEL/pytorch_model.bin
    cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin

    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3 ../code/gen_yelp_dataset_bert_task_finetune_HEADandTAIL_baseline_batch_self.py $INPUT $OUTPUT/train $MODEL $INSTANCE_1 $NUM_LABEL $RETRIVE_BATCH
    ###


    ######################################################
    ######################################################
    ####################My model##########################
    ######################################################
    ######################################################
    mkdir backup_semeval
    rm backup_semeval/*
    cp -r $OUTFILE/* backup_semeval/

    #for K in 8 16 24 32 64
    for K in 16 32 48
    do

        OUTFILE="output_pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_dt"
        rm -rf $OUTFILE
        mkdir $OUTFILE
        cp -r backup_semeval/* $OUTFILE/
        INPUT=../data/openwebtext/train.txt
        OUTPUT=../data/opendomain_finetune_noword_10000
        MODEL=bert-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
        mkdir $OUTPUT
        mkdir $MODEL
        rm -rf $MODEL/*
        cp -r bert-base-768/* $MODEL/
        rm -rf $MODEL/pytorch_model.bin
        cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin

        ###########
        #####Train a model and better retriver for generation doc representation
        ###########
        OUTFILE="output_pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_dt"


        rm -rf $OUTFILE
        mkdir $OUTFILE

        MODEL=bert-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self

        #Need to filter some sentence to train the domain model
        #Given and put some outdomain data in DATA_out
        DATA_out=../data/opendomain_finetune_noword_10000/
        DATA_in=../data/restaurant_fewshot/

        CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3 -W ignore::UserWarning ../code/pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_updateRep_opendomain_fast_entropy_self.py   --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length $MAX_LENGTH --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --augment_times 20 --K $K


        ###########
        #####Eval the model and extract the best model to _gen dir
        ###########

        MODEL=bert-base-768-yelp-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
        DATA_in=../data/restaurant_fewshot/
        DATA_out=../data/opendomain_finetune_noword_10000/
        OUTFILE="output_pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_entropy_sscl_dt"

        CUDA_VISIBLE_DEVICES=$EVAL_GPU python3 -W ignore::UserWarning ../code/eval_bert_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch_self.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length $MAX_LENGTH --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --choose_eval_test_both 2

        python3 ../code/score.py $OUTFILE


        RETRIVER_file=retriver_bert_n_result_open_domain_entropy_sscl_dt_$K"_"$ITER
        ###
        mkdir $RETRIVER_file
        rm -rf $RETRIVER_file/*

        ###########
        #####Recorde the best retriver
        ###########

        python3 ../code/extract_test_by_eval.py $OUTFILE $RETRIVER_file $N

    done


done
