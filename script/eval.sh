#!/bin/bash

#rm output_finwtune/*

#model=../../data/DKPLM_BERTbase_2layer
#model=../bert-base-uncased
#model=bert-base-uncased
model=/data5/private/suyusheng/task_selecte/bert-base-uncased

#CUDA_VISIBLE_DEVICES=0,1 python3.6 -W ignore::UserWarning code/pre_train.py   --num_labels_task 8 --do_train   --do_lower_case   --data_dir data/restaurant   --pretrain_model $model --max_seq_length 256   --train_batch_size 32 --learning_rate 2e-5   --num_train_epochs 16   --output_dir output_finwtune   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 1 --fp16


#eval
CUDA_VISIBLE_DEVICES=0 python3.6 -W ignore::UserWarning code/eval.py   --num_labels_task 8 --do_eval   --do_lower_case   --data_dir data/restaurant   --pretrain_model $model --max_seq_length 256   --eval_batch_size 8 --learning_rate 2e-5   --num_train_epochs 16   --output_dir output_finwtune   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 1 --fp16

python3 code/score.py output_finetune



