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
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification
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
tail_hidd_list = list()
#device = "cpu"
device = "cuda"


pretrained_weights = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)

fine_tuned_weight = 'roberta-base'
model = RobertaForMaskedLM.from_pretrained(pretrained_weights, output_hidden_states=True,return_dict=True)
#model.load_state_dict(torch.load(fine_tuned_weight), strict=False)

#model.to(device).half()
model.to(device)
model.eval()


old = torch.FloatTensor(768)
with open(file_in) as f:
    data = json.load(f)
    for index, d in enumerate(tqdm(data)):
        #print(d["sentence"])
        ids = tokenizer.encode(d["sentence"],add_special_tokens=True)
        #print(ids)
        ids_tail = len(ids)-1
        attention_mask = [1]*len(ids)
        #<pad> --> 1
        ids = ids+[1]*(max_length-len(ids))
        padding = [0]*(max_length-len(ids))
        attention_mask += padding

        torch_ids = torch.tensor([ids]).to(device)
        output = model(input_ids=torch_ids, attention_mask=attention_mask) #([1,100,768])
        #last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        with torch.no_grad():
            ###
            tail_hidd = output.hidden_states[0][0][ids_tail]
            #tail_hidd = output.hidden_states[0][0][0]
            ###

            ###
            #tail_hidd = output.hidden_states[0][0].mean(dim=0)
            ###


        tail_hidd = tail_hidd.to("cpu")
        all_data_dict[index] = {"sentence":d["sentence"], "aspect":d["aspect"], "sentiment":d["sentiment"], "ids":ids}
        tail_hidd_list.append(tail_hidd)

        #########
        '''
        if torch.equal(tail_hidd,old):
            for
            print(tail_hidd)
            print(old)
            exit()
            print(index)
            old = tail_hidd
        else:
            old = tail_hidd
        '''

with open(file_out+'.json', 'w') as outfile:
    json.dump(all_data_dict, outfile)
    #torch.save(all_data_dict, outfile)

#tail_hidd_tensor = torch.FloatTensor(CLS_hidd_list)
tail_hidd_tensor = torch.stack(tail_hidd_list)
print(tail_hidd_tensor.shape)
torch.save(tail_hidd_tensor, file_out+'_CLS.pt')


