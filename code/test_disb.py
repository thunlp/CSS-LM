import pickle
import json
import argparse
import logging
import random
import numpy as np
import os
import json
import sys
import time

import torch
from transformers import BertTokenizer, BertForPreTraining, BertForSequenceClassification
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

max_length=100
k=10
device="cpu"

pretrained_weights = '/data5/private/suyusheng/task_selecte/bert-base-uncased-128/'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)

fine_tuned_weight = '/data5/private/suyusheng/task_selecte/output_finetune/pytorch_model.bin_1314'
model = BertForPreTraining.from_pretrained(pretrained_weights, output_hidden_states=True,return_dict=True)
model.load_state_dict(torch.load(fine_tuned_weight), strict=False)
model.to(device)


#out_CLS = torch.load("/data5/private/suyusheng/task_selecte/data/open_domain_preprocessed/opendomain_CLS.pt")
out_CLS = torch.load("/data5/private/suyusheng/task_selecte/data/open_domain_preprocessed/opendomain_CLS_res.pt")
out_CLS = out_CLS.to(device)

#with open("/data5/private/suyusheng/task_selecte/data/open_domain_preprocessed/opendomain.json") as f:
with open("/data5/private/suyusheng/task_selecte/data/open_domain_preprocessed/opendomain_res.json") as f:
    out_data = json.load(f)

with open("../data/restaurant/train.json") as f:
    data = json.load(f)
    for index, d in enumerate(tqdm(data)):
        #if index <= 1:
        #    continue

        ids = tokenizer.encode(d["sentence"],add_special_tokens=True)
        ids = ids+[0]*(max_length-len(ids))
        torch_ids = torch.tensor([ids])
        torch_ids = torch_ids.to(device)
        output = model(torch_ids) #([1,100,768])
        CLS_hidd = output["hidden_states"][-1][0][0]
        #print(CLS_hidd.shape)
        t_start = time.time()
        result = CLS_hidd.matmul(out_CLS.reshape(out_CLS.shape[1],out_CLS.shape[0]))
        #top_n = torch.topk(result, 10, dim=None, largest=True, sorted=False, out=None)
        top_n = result.topk(k=k, dim=0, largest=True, sorted=False)
        #print(top_n)
        #print(type(top_n))
        #exit()
        bottom_n = result.topk(k=k, dim=0, largest=False, sorted=False)
        #print(bottom_n)
        #result = CLS_hidd.dot(out_CLS)
        t_end = time.time()
        print("time:",t_end-t_start)
        #print(result.shape)

        print("===Ranking===")
        print(d['sentence'])
        print(d['aspect'])
        print(d['sentiment'])
        print("===============")
        print("===============")
        print("top_n")
        for i in range(k):
            score = top_n[0][i]
            index = top_n[1][i]
            print(score)
            print(out_data[str(int(index))]['sentence'])
            print(out_data[str(int(index))]['aspect'])
            print(out_data[str(int(index))]['sentiment'])
            print("---")
        print("===============")
        print("===============")
        print("bottom_n")
        for i in range(k):
            score = bottom_n[0][i]
            index = bottom_n[1][i]
            #print(score, out_data[str(int(index))])
            print(score)
            print(out_data[str(int(index))]['sentence'])
            print(out_data[str(int(index))]['aspect'])
            print(out_data[str(int(index))]['sentiment'])
            print("---")

        #for v, id in top_n[0]:
        #    print(v,id)

        exit()

#CPU
#time: 0.018637657165527344
#time: 0.051631927490234375
#GPU
#time: 0.00010919570922851562
#time: 0.00010061264038085938
