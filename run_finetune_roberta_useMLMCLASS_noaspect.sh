#!/bin/bash
#O1 --> O2


###########################
###########################
###########################
#OUTFILE="output_finetune_roberta_yelp_DomainTask"
OUTFILE="output_finetune_roberta_yelp_DomainTask_useMLMCLASS_noaspect"
rm -rf $OUTFILE
mkdir $OUTFILE

#model=roberta-base
model=roberta-base-768-yelp-DomainTask-noword-sentiment
DATA=data/restaurant_noword
rm -rf $model
cp -r roberta-base-768 $model
cp output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect/pytorch_model.bin_30 $model/pytorch_model.bin


cp output_pretrain_roberta_including_Preprocess_DomainTask_sentiment_noaspect/pytorch_model.bin_* $OUTFILE/
#num_train_epochs 10 -->15
#CUDA_VISIBLE_DEVICES=7 python3.6 -W ignore::UserWarning code/finetune_roberta.py   --num_labels_task 8 --do_train   --do_lower_case   --data_dir $DATA   --pretrain_model $model --max_seq_length 100   --train_batch_size 32 --learning_rate 2e-5   --num_train_epochs 16   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 1 --fp16


#eval
#eval  --> can only be on one GPU
CUDA_VISIBLE_DEVICES=0 python3.6 -W ignore::UserWarning code/eval_roberta_useMLMCLASS_sentiment_noaspect.py   --num_labels_task 3 --do_eval   --do_lower_case   --data_dir $DATA   --pretrain_model $model --max_seq_length 100   --eval_batch_size 8 --learning_rate 2e-5   --num_train_epochs 16   --output_dir $OUTFILE   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 2 --fp16

python3 code/score.py $OUTFILE
