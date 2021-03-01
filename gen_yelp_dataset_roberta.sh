#!/bin/bash

#code, file_in, file_out
#restaurant
#CUDA_VISIBLE_DEVICES=5 python3 code/gen_opendomain_dataset_roberta.py data/open_domain/opendomain.json data/open_domain_preprocessed_roberta/opendomain

#yelp
CUDA_VISIBLE_DEVICES=4 python3 code/gen_yelp_dataset_roberta.py data/yelp/train.txt_all data/yelp/train
