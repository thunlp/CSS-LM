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

num_samples = 1000000

old = torch.FloatTensor(768)
with open(file_in) as f:
    #data = json.load(f)
    for index, d in tqdm(enumerate(f)):
        if index == 1000000:
            break
        if len(d) == 0:
            continue
        #print(d["sentence"])
        tokens = tokenizer.tokenize(d)
        if len(tokens)>=max_length-2:
            tokens = tokens[:max_length-2]
            tokens = ["<s>"] + tokens + ["</s>"]
            ids_tail = len(tokens)-1
        else:
            ids_tail = len(tokens)-1
            tokens = ["<s>"]+tokens+["</s>"]
        attention_mask = [1]*len(tokens)
        padding = ["<pad>"]*(max_length-len(tokens))
        tokens += padding
        attention_mask += [0]*len(padding)


        ids = tokenizer.encode(tokens, add_special_tokens=False)
        torch_ids = torch.tensor([ids]).to(device)
        attention_mask = torch.tensor([attention_mask]).to(device)
        output = model(input_ids=torch_ids, attention_mask=attention_mask) #([1,100,768])
        #last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        with torch.no_grad():
            #print(output.hidden_states[0])
            #print("--")
            #print(output.hidden_states)
            #print(output.hidden_states.shape)
            #exit()
            #print(output.hidden_states[:])
            #print("---")
            #print(len(output.hidden_states[:]))
            #every <s> in each layer
            tail_hidd = [x[0] for x in output.hidden_states[:]]
            tail_hidd = torch.stack(tail_hidd)
            tail_hidd = tail_hidd[:,0,:]
            ###
            #tail_hidd = output.hidden_states[0][0][ids_tail]
            #tail_hidd = output.hidden_states[0][0][0]
            #tail_hidd = output.hidden_states[0][0][:]
            #tail_hidd = (output.hidden_states[0][0][:]).mean(dim=0)
            #print(tail_hidd.shape)
            #exit()
            ###

            ###
            #tail_hidd = output.hidden_states[0][0].mean(dim=0)
            ###

        #print(tail_hidd)
        #print(tail_hidd.shape)
        #exit()
        tail_hidd = tail_hidd.to("cpu")
        #all_data_dict[index] = {"sentence":d["sentence"], "aspect":d["aspect"], "sentiment":d["sentiment"], "ids":ids}
        all_data_dict[index] = {"sentence":d}
        tail_hidd_list.append(tail_hidd)

        #########
        if torch.equal(tail_hidd,old):
            #print(tail_hidd)
            #print(old)
            print(index)
            print("------")
            old = tail_hidd
        else:
            old = tail_hidd

with open(file_out+'.json', 'w') as outfile:
    json.dump(all_data_dict, outfile)
    #torch.save(all_data_dict, outfile)

#tail_hidd_tensor = torch.FloatTensor(CLS_hidd_list)
tail_hidd_tensor = torch.stack(tail_hidd_list)
print(tail_hidd_tensor.shape)
torch.save(tail_hidd_tensor, file_out+'_CLS.pt')


