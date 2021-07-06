#!/bin/bash
#level O2 (smaller)

rm output_create_pretrain_data/*

#model=../../data/DKPLM_BERTbase_2layer
#model=../bert-base-uncased
model=bert-base-uncased
#model=/data5/private/suyusheng/task_selecte/bert-base-uncased-128

#num_train_epochs 10 -->15
CUDA_VISIBLE_DEVICES=6,7 python3.6 -W ignore::UserWarning code/create_pretrain_data.py   --num_labels_task 8 --do_train   --do_lower_case   --data_dir data/restaurant/train.txt   --pretrain_model $model --max_seq_length 100   --train_batch_size 16 --learning_rate 2e-5   --num_train_epochs 3   --output_dir output_create_pretrain_data  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16


