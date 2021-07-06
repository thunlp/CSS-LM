#!/bin/bash
#fix
###
#FINAL_file=final_n_result_open_domain_scicite_entropy_sscl_dt
#RETRIVER_file=retriver_n_result_open_domain_scicite_entropy_sscl_dt
#BASELINE_file=baseline_n_result_open_domain_scicite_entropy_sscl_dt
FINAL_file=final_bert_n_result_open_domain_scicite_entropy_sscl_dt
RETRIVER_file=retriver_bert_n_result_open_domain_scicite_entropy_sscl_dt
BASELINE_file=baseline_bert_n_result_open_domain_scicite_entropy_sscl_dt
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
NUM_LABEL=3
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


    mkdir ../data/scicite_fewshot
    cd ../data/scicite_fewshot
    #python3 extract_instance.py train_all.json $N
    cp train.json_$N train.json
    #cp train_$N.json train.json
    cd ../../script

    ###########
    ###Train Baseline models
    ###########
    #fix
    ###
    OUTFILE="output_pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_scicite_entropy_sscl_dt"
    ###
    rm -rf $OUTFILE
    mkdir $OUTFILE

    #fix
    ###
    #MODEL=roberta-base
    MODEL=bert-base-uncased
    ###
    DATA_in=../data/scicite_fewshot/
    DATA_out=../data/opendomain_finetune_scicite_10000/

    #fix
    ###
    CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning ../code/init_bert_sscl_dt.py  --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K 16
    ###


    ###########
    #####Eval Baseline models
    ###########
    #fix
    ###
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning ../code/eval_bert_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch_self.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16 --choose_eval_test_both 2
    ###


    python3 ../code/score.py $OUTFILE


    ###########
    #####Recorde the best retriver: Baseline
    ###########
    echo $BASELINE_file"_"${ITER}
    mkdir $BASELINE_file"_"${ITER}
    rm $BASELINE_file"_"${ITER}/*
    #python3 code/extract.py $OUTFILE $BASELINE_file $N test
    python3 ../code/extract_test_by_eval.py $OUTFILE $BASELINE_file"_"${ITER} $N


    ##########
    ###Initial retrive (First epoch retrive): Assume: pytorch_model.bin is retriver
    #####Retrive 10000 data
    #########
    INPUT=../data/openwebtext/train.txt
    OUTPUT=../data/opendomain_finetune_scicite_10000
    #fix
    ###
    #MODEL=roberta-base-768-scicite-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
    MODEL=bert-base-768-scicite-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
    ###
    mkdir $OUTPUT
    mkdir $MODEL
    rm -rf $MODEL/*
    #fix
    ###
    #cp -r roberta-base-768/* $MODEL/
    cp -r bert-base-768/* $MODEL/
    ###
    rm -rf $MODEL/pytorch_model.bin
    cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin
    #cp $OUTFILE/pytorch_model.bin_test_best $MODEL/pytorch_model.bin

    #fix
    ###
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python3 ../code/gen_yelp_dataset_bert_task_finetune_HEADandTAIL_baseline_batch_self.py $INPUT $OUTPUT/train $MODEL $INSTANCE_1 $NUM_LABEL $RETRIVE_BATCH
    ###


    ######################################################
    ######################################################
    ####################My model##########################
    ######################################################
    ######################################################
    #####
    #rm output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_scicite_entropy_st/*
    #cp -r $OUTFILE/* output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_scicite_entropy_st/
    #####
    mkdir backup_scicite
    rm backup_scicite/*
    cp -r $OUTFILE/* backup_scicite/
    #for K in 8 16 32 64 128
    #for K in 8 16 24 32 #64
    for K in 16 32 48
    #for K in 16
    do
        #fix
        ###
        #OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_scicite_entropy_sscl_dt"
        OUTFILE="output_pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_scicite_entropy_sscl_dt"
        ###
        ###
        rm -rf $OUTFILE
        mkdir $OUTFILE
        cp -r backup_scicite/* $OUTFILE/
        ###
        INPUT=../data/openwebtext/train.txt
        OUTPUT=../data/opendomain_finetune_scicite_10000
        #fix
        ###
        #MODEL=roberta-base-768-scicite-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
        MODEL=bert-base-768-scicite-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
        ###
        mkdir $OUTPUT
        mkdir $MODEL
        rm -rf $MODEL/*
        #fix
        ###
        #cp -r roberta-base-768/* $MODEL/
        cp -r bert-base-768/* $MODEL/
        ###
        rm -rf $MODEL/pytorch_model.bin
        cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin
        #cp $OUTFILE/pytorch_model.bin_test_best $MODEL/pytorch_model.bin
        ###
        ###########
        #####Train a model and better retriver for generation doc representation
        ###########
        #fix
        ###
        #OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_scicite_entropy_sscl_dt"
        OUTFILE="output_pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_opendomain_scicite_entropy_sscl_dt"
        ###
        rm -rf $OUTFILE
        mkdir $OUTFILE

        ##From org train
        #MODEL=roberta-base
        #From trained <s> train
        #fix
        ###
        #MODEL=roberta-base-768-scicite-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
        MODEL=bert-base-768-scicite-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
        ###

        #Need to filter some sentence to train the domain model
        #Given and put some outdomain data in DATA_out
        DATA_out=../data/opendomain_finetune_scicite_10000/
        DATA_in=../data/scicite_fewshot/
        #fix
        ###

        CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3.6 -W ignore::UserWarning ../code/pretrain_bert_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL_updateRep_opendomain_fast_entropy_self.py   --num_labels_task $NUM_LABEL --do_train   --do_lower_case   --data_dir_outdomain $DATA_out  --data_dir_indomain $DATA_in --pretrain_model $MODEL --max_seq_length 100   --train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16 --augment_times 20 --K $K
        ###



        ###########
        #####Eval the model and extract the best model to _gen dir
        ###########
        #fix
        ###

        #MODEL=roberta-base-768-scicite-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
        MODEL=bert-base-768-scicite-DomainTask-noword-sentiment-HEADandTAIL-opendomain-entropy-self
        ###
        DATA_in=../data/scicite_fewshot/
        #rm -rf $MODEL
        #cp -r roberta-base-768 $MODEL
        #rm $MODEL/pytorch_model.bin
        #cp $OUTFILE/pytorch_model.bin_dev_best $MODEL/pytorch_model.bin
        #OUTFILE="output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect_HEADandTAIL"

        #eval
        #eval  --> can only be on one GPU
        ###Extract and eval  model
        #fix
        ###
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python3.6 -W ignore::UserWarning ../code/eval_bert_useMLMCLASS_sentiment_noaspect_HEADandTAIL_updateRep_batch_self.py   --num_labels_task $NUM_LABEL --do_eval   --do_lower_case   --data_dir $DATA_in   --pretrain_model $MODEL --max_seq_length 100   --eval_batch_size $BATCH_SIZE_EVAL --learning_rate $LEARNING_RATE   --num_train_epochs $N_times_1   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16 --choose_eval_test_both 2
        ###

        python3 ../code/score.py $OUTFILE

        ###
        #fix
        ###
        #RETRIVER_file=retriver_n_result_open_domain_scicite_entropy_sscl_dt_$K"_"$ITER
        RETRIVER_file=retriver_bert_n_result_open_domain_scicite_entropy_sscl_dt_$K"_"$ITER
        ###
        mkdir $RETRIVER_file
        rm -rf $RETRIVER_file/*
        ###


        ###########
        #####Recorde the best retriver
        ###########
        #python3 code/extract.py $OUTFILE $RETRIVER_file $N eval
        #python3 code/extract.py $OUTFILE $RETRIVER_file $N test
        python3 ../code/extract_test_by_eval.py $OUTFILE $RETRIVER_file $N
    done

done
