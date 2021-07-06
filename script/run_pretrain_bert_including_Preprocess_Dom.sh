#!/bin/bash
#level O2 (smaller)

rm output_pretrain_bert_including_Preprocess_Dom/*

model=bert-base-uncased
#DATA=data/restaurant/train.txt
DATA=data/open_domain_preprocessed_bert/opendomain_tr.txt

#num_train_epochs 10 -->15
CUDA_VISIBLE_DEVICES=6 python3.6 -W ignore::UserWarning code/pretrain_bert_including_Preprocess_Dom.py   --num_labels_task 8 --do_train   --do_lower_case   --data_dir $DATA   --pretrain_model $model --max_seq_length 100   --train_batch_size 4 --learning_rate 2e-5   --num_train_epochs 3   --output_dir output_pretrain_bert_including_Preprocess_Dom  --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 0 --fp16


