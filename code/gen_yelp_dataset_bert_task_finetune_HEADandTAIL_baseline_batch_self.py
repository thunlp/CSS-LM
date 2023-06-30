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
from transformers import BertTokenizer, BertForMaskedLM
from transformers.modeling_bert_updateRep_self import BertForMaskedLMDomainTask, BertForSequenceClassification
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


file_in = sys.argv[1]
file_out = sys.argv[2]
model = sys.argv[3]
num_samples = int(sys.argv[4])
num_labels = int(sys.argv[5])
batch_size = int(sys.argv[6])
#batch_size = 16*3
#batch_size = 2
device = "cuda"
max_length = 100
local_rank = -1
no_cuda = False
seed = 42
opt_level='O1'
fp16=False
learning_rate=5e-5
adam_epsilon=1e-8
weight_decay=0.0




def load_data():
    with open(file_in) as f:
        #all_list = list()
        print("Loading data:")
        all_torch_ids = list()
        all_attention_mask = list()
        counter=0
        for index, d in tqdm(enumerate(f)):
            #if index == int(num_samples):
            if counter >= int(num_samples):
                break
            if len(d) == 0:
                continue
            tokens = tokenizer.tokenize(d)
            #####
            if len(tokens) < 15:
                continue
            else:
                counter+=1
            #####
            ids_tail = 0
            if len(tokens)>=max_length-2:
                tokens = tokens[:max_length-2]
                #tokens = ["<s>"] + tokens + ["</s>"]
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                ids_tail = len(tokens)-1
            else:
                ids_tail = len(tokens)-1
                tokens = ["[CLS]"]+tokens+["[SEP]"]
            attention_mask = [1]*len(tokens)
            padding = ["<pad>"]*(max_length-len(tokens))
            tokens += padding
            attention_mask += [0]*len(padding)

            ids = tokenizer.encode(tokens, add_special_tokens=False)
            torch_ids = torch.tensor(ids)
            attention_mask = torch.tensor(attention_mask)

            all_torch_ids.append(torch_ids)
            all_attention_mask.append(attention_mask)

        ###
        #cur_tensor = (torch_ids, attention_mask)
        #all_list.append(cur_tensor)
        all_torch_ids = torch.stack(all_torch_ids)
        all_attention_mask = torch.stack(all_attention_mask)
        all_list = TensorDataset(all_torch_ids, all_attention_mask)
        ###
    return all_list



################
if local_rank == -1 or no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    print("===")
    print(device)
    print(n_gpu)
    print("===")
else:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

tokenizer = BertTokenizer.from_pretrained(model)


# Prepare model
model = BertForMaskedLMDomainTask.from_pretrained(model, output_hidden_states=True, return_dict=True, num_labels=num_labels)
model.to(device)

##########################
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#no_decay = ['bias', 'LayerNorm.weight']
no_grad = ['bert.encoder.layer.11.output.dense_ent', 'bert.encoder.layer.11.output.LayerNorm_ent']
param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_grad)]
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
##########################

if fp16:
    try:
        from apex import amp
    except:
        print("ERROR: Have mo apex")
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)

if local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


all_sentence = load_data()
if local_rank == -1:
    train_sampler = RandomSampler(all_sentence)
else:
    train_sampler = DistributedSampler(all_sentence)
train_dataloader = DataLoader(all_sentence, sampler=train_sampler, batch_size=batch_size)
###############


old = torch.FloatTensor(768)
docs_tail = list()
docs_head = list()
id_doc_tensor = list()
counter_domain = 0
counter_task = 0
for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    with torch.no_grad():
        batch = tuple(t.to(device) for t in batch)
        input_ids_org_, attention_mask = batch
        in_task_rep, in_domain_rep = model(input_ids_org=input_ids_org_, attention_mask=attention_mask, func="in_domain_task_rep") #([1,100,768])

        in_task_rep= in_task_rep.to("cpu")
        in_domain_rep = in_domain_rep.to("cpu")
        input_ids_org_ = input_ids_org_.to("cpu")

        #####
        #####
        if step==0:
            docs_tail.append(in_domain_rep)
            docs_head.append(in_task_rep)
            id_doc_tensor.append(input_ids_org_)
        else:
            if docs_tail[-1].shape[0] == in_domain_rep.shape[0]:
                docs_tail.append(in_domain_rep)
                docs_head.append(in_task_rep)
                id_doc_tensor.append(input_ids_org_)
            else:
                rest_tail = in_domain_rep
                rest_head = in_task_rep
                rest_input = input_ids_org_

        counter_domain += int(in_domain_rep.shape[0])
        counter_task += int(in_task_rep.shape[0])

        if counter_domain!=counter_task:
            print("Error")
            exit()

###
docs_tail = torch.stack(docs_tail)
docs_tail = docs_tail.reshape(docs_tail.shape[0]*docs_tail.shape[1],1,docs_tail.shape[-1])
try:
    rest_tail = rest_tail.reshape(rest_tail.shape[0],1,rest_tail.shape[-1])
    docs_tail = torch.cat([docs_tail,rest_tail],0)
except:
    pass


docs_head = torch.stack(docs_head)
docs_head = docs_head.reshape(docs_head.shape[0]*docs_head.shape[1],1,docs_head.shape[-1])
try:
    rest_head = rest_head.reshape(rest_head.shape[0],1,rest_head.shape[-1])
    docs_head = torch.cat([docs_head,rest_head],0)
except:
    pass


id_doc_tensor = torch.stack(id_doc_tensor)
id_doc_tensor = id_doc_tensor.reshape(id_doc_tensor.shape[0]*id_doc_tensor.shape[1],id_doc_tensor.shape[-1])
try:
    rest_input = rest_input.reshape(rest_input.shape[0],rest_input.shape[-1])
    id_doc_tensor = torch.cat([id_doc_tensor,rest_input],0)
except:
    pass


docs_tail = docs_tail.to("cpu")
docs_head = docs_head.to("cpu")
id_doc_tensor = id_doc_tensor.to("cpu")


all_data_dict = dict()
head_hidd_list = list()
tail_hidd_list = list()
for id_, index in tqdm(enumerate(id_doc_tensor)):
    #all_data_dict[id_] = {"sentence":tokenizer.decode(id_doc_tensor[id_]).replace("<pad>","").replace("<s>","").replace("</s>","").replace("\n","").replace("\u0000","")}
    all_data_dict[id_] = {"sentence":tokenizer.decode(id_doc_tensor[id_]).replace("<pad>","").replace("[CLS]","").replace("[SEP]","").replace("\n","").replace("\u0000","")}
    head_hidd_list.append(docs_head[id_])
    tail_hidd_list.append(docs_tail[id_])

with open(file_out+'.json', 'w') as outfile:
    json.dump(all_data_dict, outfile)

head_hidd_tensor = torch.stack(head_hidd_list)
tail_hidd_tensor = torch.stack(tail_hidd_list)

torch.save(head_hidd_tensor, file_out+'_head.pt')
torch.save(tail_hidd_tensor, file_out+'_tail.pt')


