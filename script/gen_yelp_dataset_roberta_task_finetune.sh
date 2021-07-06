#!/bin/bash

#code, file_in, file_out
#restaurant
#CUDA_VISIBLE_DEVICES=5 python3 code/gen_opendomain_dataset_roberta.py data/open_domain/opendomain.json data/open_domain_preprocessed_roberta/opendomain

'''
cp roberta-base-768/* roberta-base-768-task-finetune/
cp output_finetune_roberta/pytorch_model.bin roberta-bash-768-task-finetune/pytorch_model.bin
'''

INPUT=data/yelp/train.txt_all
#OUTPUT=data/yelp_finetune/train
OUTPUT=data/yelp_finetune_noword_10000/train
#model=roberta-base-this
MODEL=roberta-base-768-yelp-DomainTask-noword-sentiment
INSTANCE=10000

#yelp
#CUDA_VISIBLE_DEVICES=0 python3 code/gen_yelp_dataset_roberta_task_finetune.py data/yelp/train.txt_all data/yelp_finetune/train $model $instance

CUDA_VISIBLE_DEVICES=0 python3 code/gen_yelp_dataset_roberta_task_finetune.py data/yelp/train.txt_all data/yelp_finetune_noword_10000/train $MODEL $INSTANCE
