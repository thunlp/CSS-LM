import pickle
import json
import argparse
import logging
import random
import numpy as np
import os
import json
import sys

import torch
from transformers import BertTokenizer, BertForPreTraining, BertForSequenceClassification
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

#with open(FILE) as f:
#    file = pickle.load(f)

file_in = sys.argv[1]
file_out = sys.argv[2]

all_data_dict = dict()
max_length = 100
CLS_hidd_list = list()
#device = "cpu"
device = "cuda"


pretrained_weights = '/data5/private/suyusheng/task_selecte/bert-base-uncased-768/'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)

#fine_tuned_weight = '/data5/private/suyusheng/task_selecte/output_finetune/pytorch_model.bin_1314'
model = BertForPreTraining.from_pretrained(pretrained_weights, output_hidden_states=True,return_dict=True)
#model.load_state_dict(torch.load(fine_tuned_weight), strict=False)

#model.to(device).half()
model.to(device)


old = torch.FloatTensor(768)
with open(file_in) as f:
    data = json.load(f)
    for index, d in enumerate(tqdm(data)):
        ids = tokenizer.encode(d["sentence"],add_special_tokens=True)
        ids = ids+[0]*(max_length-len(ids))
        torch_ids = torch.tensor([ids]).to(device)
        output = model(torch_ids) #([1,100,768])
        with torch.no_grad():
            CLS_hidd = output["hidden_states"][-1][0][0]
        CLS_hidd = CLS_hidd.to("cpu")
        all_data_dict[index] = {"sentence":d["sentence"], "aspect":d["aspect"], "sentiment":d["sentiment"], "ids":ids}
        CLS_hidd_list.append(CLS_hidd)

        ###
        '''
        if torch.equal(CLS_hidd,old):
            print(index)
            old = CLS_hidd
        else:
            old = CLS_hidd
        '''

with open(file_out+'.json', 'w') as outfile:
    json.dump(all_data_dict, outfile)
    #torch.save(all_data_dict, outfile)

#CLS_hidd_tensor = torch.FloatTensor(CLS_hidd_list)
CLS_hidd_tensor = torch.stack(CLS_hidd_list)
print(CLS_hidd_tensor.shape)
torch.save(CLS_hidd_tensor, file_out+'_CLS.pt')


