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
#from transformers.modeling_roberta import RobertaForMaskedLMDomainTask
from transformers.modeling_roberta_updateRep import RobertaForMaskedLMDomainTask
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

#with open(FILE) as f:
#    file = pickle.load(f)

file_in = sys.argv[1]
file_out = sys.argv[2]
model = sys.argv[3]
num_samples = int(sys.argv[4])
#num_samples = 1000000

all_data_dict = dict()
max_length = 100
head_hidd_list = list()
tail_hidd_list = list()
#device = "cpu"
device = "cuda"


pretrained_weights = model
tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)

fine_tuned_weight = model
#model = RobertaForMaskedLM.from_pretrained(pretrained_weights, output_hidden_states=True,return_dict=True)
model = RobertaForMaskedLMDomainTask.from_pretrained(pretrained_weights, output_hidden_states=True,return_dict=True,num_labels=3)
#model.load_state_dict(torch.load(fine_tuned_weight), strict=False)

#model.to(device).half()
model.to(device)
model.eval()

#num_samples = 1000000

old = torch.FloatTensor(768)
with open(file_in) as f:
    #data = json.load(f)
    for index, d in tqdm(enumerate(f)):
        #print(type(index),index)
        #if index == 1000000:
        if index == int(num_samples):
            break
        if len(d) == 0:
            continue
        #print(d["sentence"])
        tokens = tokenizer.tokenize(d)
        ids_tail = 0
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
        #output = model(input_ids=torch_ids , attention_mask=attention_mask, func="g) #([1,100,768])
        output = model(input_ids_org=torch_ids , attention_mask=attention_mask, func="gen_rep") #([1,100,768])
        #output = model(input_ids=torch_ids, attention_mask=attention_mask, func="in_domain_task_rep") #([1,100,768])
        #last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        with torch.no_grad():
            '''
            #every <s> in each layer
            head_hidd = [x[0] for x in output.hidden_states[:]]
            head_hidd = torch.stack(head_hidd)
            head_hidd = head_hidd[:,0,:]
            '''

            #Only use last layer <s> and </s>
            head_hidd = output.last_hidden_state[:,0,:]
            tail_hidd = output.last_hidden_state[:,ids_tail,:]

        head_hidd = head_hidd.to("cpu")
        tail_hidd = tail_hidd.to("cpu")
        #all_data_dict[index] = {"sentence":d["sentence"], "aspect":d["aspect"], "sentiment":d["sentiment"], "ids":ids}
        all_data_dict[index] = {"sentence":d}
        head_hidd_list.append(head_hidd)
        tail_hidd_list.append(tail_hidd)

        #########
        '''
        if torch.equal(head_hidd,old):
            #print(tail_hidd)
            #print(old)
            print(index)
            print("------")
            old = head_hidd
        else:
            old = head_hidd
        '''

with open(file_out+'.json', 'w') as outfile:
    json.dump(all_data_dict, outfile)
    #torch.save(all_data_dict, outfile)

#tail_hidd_tensor = torch.FloatTensor(CLS_hidd_list)
head_hidd_tensor = torch.stack(head_hidd_list)
tail_hidd_tensor = torch.stack(tail_hidd_list)
#print(head_hidd_tensor.shape)
#print(tail_hidd_tensor.shape)

torch.save(head_hidd_tensor, file_out+'_head.pt')
torch.save(tail_hidd_tensor, file_out+'_tail.pt')


