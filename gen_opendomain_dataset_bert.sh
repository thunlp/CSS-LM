#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python3 code/gen_opendomain_dataset_bert.py data/open_domain/opendomain.json data/open_domain_preprocessed_bert/opendomain
