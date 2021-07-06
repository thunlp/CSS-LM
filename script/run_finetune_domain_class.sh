#!/bin/bash
#O1 --> O2

rm output_finetune_domain_class/*

#model=../../data/DKPLM_BERTbase_2layer
#model=../bert-base-uncased
#model=bert-base-uncased
model=/data5/private/suyusheng/task_selecte/bert-base-uncased-128

#num_train_epochs 10 -->15
CUDA_VISIBLE_DEVICES=0,1 python3.6 -W ignore::UserWarning code/finetune_bert.py   --num_labels_task 8 --do_train   --do_lower_case   --data_dir data/res_lap_domain_class   --pretrain_model $model --max_seq_length 100   --train_batch_size 32 --learning_rate 2e-5   --num_train_epochs 9   --output_dir output_finetune_domain_class   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 1 --fp16


#eval
#eval  --> can only be on one GPU
CUDA_VISIBLE_DEVICES=6,7 python3.6 -W ignore::UserWarning code/eval_bert.py   --num_labels_task 8 --do_eval   --do_lower_case   --data_dir data/res_lap_domain_class   --pretrain_model $model --max_seq_length 100   --eval_batch_size 8 --learning_rate 2e-5   --num_train_epochs 9   --output_dir output_finetune_domain_class   --loss_scale 128 --weight_decay 0 --adam_epsilon 1e-8 --max_grad_norm 1 --fp16_opt_level O1 --task 1 --fp16

python3 code/score.py output_finetune_domain_class

