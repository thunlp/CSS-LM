#!/bin/bash
#fix
###
#FINAL_file=final_n_result_open_domain_aclintent_entropy_finetune
#RETRIVER_file=retriver_n_result_open_domain_aclintent_entropy_finetune
#BASELINE_file=baseline_n_result_open_domain_aclintent_entropy_finetune
FINAL_file=final_bert_n_result_open_domain_aclintent_entropy_finetune
RETRIVER_file=retriver_bert_n_result_open_domain_aclintent_entropy_finetune
BASELINE_file=baseline_bert_n_result_open_domain_aclintent_entropy_finetune
###
mkdir $FINAL_file
mkdir $RETRIVER_file
mkdir $BASELINE_file
rm -rf $FINAL_file/*
rm -rf $RETRIVER_file/*
rm -rf $BASELINE_file/*

#BATCH_SIZE=16
RETRIVE_BATCH=1024
#BATCH_SIZE=4
BATCH_SIZE=${10}
BATCH_SIZE_EVAL=180
NUM_LABEL=6
#TRAIN_GPU=3,4,5,6,7
#EVAL_GPU=3,4,5,6,7
#TRAIN_GPU=0,1,2,5
#EVAL_GPU=0,1,2,5
#TRAIN_GPU=4,5,6,7
#EVAL_GPU=4,5,6,7
#TRAIN_GPU=0,1,2,3
#EVAL_GPU=0,1,2,3
TRAIN_GPU=$1,$2,$3,$4
EVAL_GPU=$1,$2,$3,$4
#TRAIN_GPU=1,2,5,7
#EVAL_GPU=1,2,5,7
#INSTANCE_1=36603
INSTANCE_1=40000
INSTANCE_2=10000
#INSTANCE=100

#1, 10 --> 62...

#8 16 24
#for N in 1 5 10 20 30 40 50
#for N in 10 20 30 40 50
ITER=${12}
for N in $5
#for N in 25 50 100
#for N in "all"
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


    mkdir data/aclintent_fewshot
    cd data/aclintent_fewshot
    #python3 extract_instance.py train_all.json $N
    cp train.json_$N train.json
    #cp train_$N.json train.json
    cd ../..

    ###########
    ###Train Baseline models
    ###########
    #fix
    ###
    #OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_aclintent_entropy_finetune"
    OUTFILE="output_pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_aclintent_entropy_finetune_all"
    ###
    rm -rf $OUTFILE
    mkdir $OUTFILE

    #fix
    ###
    #MODEL=roberta-base
    MODEL=bert-base-uncased
    ###
    DATA_in=data/aclintent_fewshot/
    DATA_out=data/opendomain_finetune_aclintent_10000_all/

    #fix
    ###
    #CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/finetune_roberta_sentiment_class_noaspect_self.py   --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16
    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning code/finetune_bert_sentiment_class_noaspect_self.py   --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16
    ###


    ###########
    #####Eval Baseline models
    ###########
    #fix
    ###
    #CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/eval_roberta_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch_self.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16 --choose_eval_test_both 2
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning code/eval_bert_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch_self.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16 --choose_eval_test_both 2
    ###


    python3 code/score.py $OUTFILE


    ###########
    #####Recorde the best retriver: Baseline
    ###########
    python3 code/extract.py $OUTFILE $BASELINE_file $N test
    python3 code/extract.py $OUTFILE $BASELINE_file $N eval
    mkdir $BASELINE_file"_"${ITER}
    rm $BASELINE_file"_"${ITER}/*
    python3 code/extract_test_by_eval.py $OUTFILE $BASELINE_file"_"${ITER} $N


done
