# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random
from io import open
import json
import time
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss

from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification
#from transformers.modeling_roberta import RobertaForMaskedLMDomainTask
from transformers.modeling_roberta_updateRep_self import RobertaForMaskedLMDomainTask
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


ce_loss = torch.nn.CrossEntropyLoss(reduction='none')


def get_parameter(parser):

    ## Required parameters
    parser.add_argument("--data_dir_indomain",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus.(In Domain)")
    parser.add_argument("--data_dir_outdomain",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus.(Out Domain)")
    parser.add_argument("--pretrain_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--augment_times",
                        default=None,
                        type=int,
                        required=True,
                        help="Default batch_size/augment_times to save model")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    ####
    parser.add_argument("--num_labels_task",
                        default=None, type=int,
                        required=True,
                        help="num_labels_task")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument('--fp16_opt_level',
                        type=str,
                        default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--task",
                        default=0,
                        type=int,
                        required=True,
                        help="Choose Task")
    parser.add_argument("--K",
                        default=None,
                        type=int,
                        required=True,
                        help="K size")
    ####
    return parser


def return_Classifier(weight, bias, dim_in, dim_out):
    #LeakyReLU = torch.nn.LeakyReLU
    classifier = torch.nn.Linear(dim_in, dim_out , bias=True)
    #print(classifier)
    #print(classifier.weight)
    #print(classifier.weight.shape)
    #print(classifier.weight.data)
    #print(classifier.weight.data.shape)
    #print("---")
    classifier.weight.data = weight.to("cpu")
    classifier.bias.data = bias.to("cpu")
    classifier.requires_grad=False
    #print(classifier)
    #print(classifier.weight)
    #print(classifier.weight.shape)
    #print("---")
    #exit()
    #print(classifier)
    #exit()
    #logit = LeakyReLU(classifier)
    return classifier


def load_GeneralDomain(dir_data_out):

    ###
    print("===========")
    print("Load CLS.pt and train.json")
    print("-----------")
    docs_head = torch.load(dir_data_out+"train_head.pt")
    docs_tail = torch.load(dir_data_out+"train_tail.pt")
    print("CLS.pt Done")
    print(docs_head.shape)
    print(docs_tail.shape)
    print("-----------")
    with open(dir_data_out+"train.json") as file:
        data = json.load(file)
    print("train.json Done")
    print("===========")
    docs_tail_head = torch.cat([docs_tail, docs_head],2)
    return docs_tail_head, docs_head, docs_tail, data
    ###


parser = argparse.ArgumentParser()
parser = get_parameter(parser)
args = parser.parse_args()
#print(args.data_dir_outdomain)
#exit()

docs_tail_head, docs_head, docs_tail, data = load_GeneralDomain(args.data_dir_outdomain)
######
if docs_head.shape[1]!=1: #UnboundLocalError: local variable 'docs' referenced before assignment
    #last <s>
    #docs = docs[:,0,:].unsqueeze(1)
    #mean 13 layers <s>
    docs_head = docs_head.mean(1).unsqueeze(1)
    print(docs_head.shape)
else:
    print(docs_head.shape)
if docs_tail.shape[1]!=1: #UnboundLocalError: local variable 'docs' referenced before assignment
    #last <s>
    #docs = docs[:,0,:].unsqueeze(1)
    #mean 13 layers <s>
    docs_tail = docs_tail.mean(1).unsqueeze(1)
    print(docs_tail.shape)
else:
    print(docs_tail.shape)
######

def in_Domain_Task_Data_mutiple(data_dir_indomain, tokenizer, max_seq_length):
    ###Open
    with open(data_dir_indomain+"train.json") as file:
        data = json.load(file)

    ###Preprocess
    num_label_list = list()
    label_sentence_dict = dict()
    num_sentiment_label_list = list()
    sentiment_label_dict = dict()
    for line in data:
        #line["sentence"]
        #line["aspect"]
        #line["sentiment"]
        num_sentiment_label_list.append(line["sentiment"])
        #num_label_list.append(line["aspect"])
        num_label_list.append(line["sentiment"])

    num_label = sorted(list(set(num_label_list)))
    label_map = {label : i for i , label in enumerate(num_label)}
    num_sentiment_label = sorted(list(set(num_sentiment_label_list)))
    sentiment_label_map = {label : i for i , label in enumerate(num_sentiment_label)}
    print("=======")
    print("label_map:")
    print(label_map)
    print("=======")
    print("=======")
    print("sentiment_label_map:")
    print(sentiment_label_map)
    print("=======")

    ###Create data: 1 choosed data along with the rest of 7 class data

    '''
    all_input_ids = list()
    all_input_mask = list()
    all_segment_ids = list()
    all_lm_labels_ids = list()
    all_is_next = list()
    all_tail_idxs = list()
    all_sentence_labels = list()
    '''
    cur_tensors_list = list()
    #print(list(label_map.values()))
    candidate_label_list = list(label_map.values())
    candidate_sentiment_label_list = list(sentiment_label_map.values())
    all_type_sentence = [0]*len(candidate_label_list)
    all_type_sentiment_sentence = [0]*len(candidate_sentiment_label_list)
    for line in data:
        #line["sentence"]
        #line["aspect"]
        sentiment = line["sentiment"]
        sentence = line["sentence"]
        #label = line["aspect"]
        label = line["sentiment"]


        tokens_a = tokenizer.tokenize(sentence)
        #input_ids = tokenizer.encode(sentence, add_special_tokens=False)
        '''
        if "</s>" in tokens_a:
            print("Have more than 1 </s>")
            #tokens_a[tokens_a.index("<s>")] = "s"
            for i in range(len(tokens_a)):
                if tokens_a[i] == "</s>":
                    tokens_a[i] == "s"
        '''


        # tokenize
        cur_example = InputExample(guid=id, tokens_a=tokens_a, tokens_b=None, is_next=0)
        # transform sample to features
        cur_features = convert_example_to_features(cur_example, max_seq_length, tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_ids_org),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(0),
                       torch.tensor(cur_features.tail_idxs),
                       torch.tensor(label_map[label]),
                       torch.tensor(sentiment_label_map[sentiment]))

        cur_tensors_list.append(cur_tensors)

        ###
        if label_map[label] in candidate_label_list:
            all_type_sentence[label_map[label]]=cur_tensors
            candidate_label_list.remove(label_map[label])

        if sentiment_label_map[sentiment] in candidate_sentiment_label_list:
            #print("----")
            #print(sentiment_label_map[sentiment])
            #print("----")
            all_type_sentiment_sentence[sentiment_label_map[sentiment]]=cur_tensors
            candidate_sentiment_label_list.remove(sentiment_label_map[sentiment])
        ###




    return all_type_sentiment_sentence, cur_tensors_list



def AugmentationData_Domain(bottom_k, top_k, tokenizer, max_seq_length):
    #top_k_shape = top_k.indices.shape
    #ids = top_k.indices.reshape(top_k_shape[0]*top_k_shape[1]).tolist()
    top_k_shape = top_k["indices"].shape
    ids_pos = top_k["indices"].reshape(top_k_shape[0]*top_k_shape[1]).tolist()
    #ids = top_k["indices"]

    bottom_k_shape = bottom_k["indices"].shape
    ids_neg = bottom_k["indices"].reshape(bottom_k_shape[0]*bottom_k_shape[1]).tolist()

    #print(ids_pos)
    #print(ids_neg)
    #exit()

    ids = ids_pos+ids_neg


    all_input_ids = list()
    all_input_ids_org = list()
    all_input_mask = list()
    all_segment_ids = list()
    all_lm_labels_ids = list()
    all_is_next = list()
    all_tail_idxs = list()
    all_id_domain = list()

    for id, i in enumerate(ids):
        t1 = data[str(i)]['sentence']

        #tokens_a = tokenizer.tokenize(t1)
        tokens_a = tokenizer.tokenize(t1)
        '''
        if "</s>" in tokens_a:
            print("Have more than 1 </s>")
            #tokens_a[tokens_a.index("<s>")] = "s"
            for i in range(len(tokens_a)):
                if tokens_a[i] == "</s>":
                    tokens_a[i] = "s"
        '''

        # tokenize
        cur_example = InputExample(guid=id, tokens_a=tokens_a, tokens_b=None, is_next=0)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, max_seq_length, tokenizer)

        all_input_ids.append(torch.tensor(cur_features.input_ids))
        all_input_ids_org.append(torch.tensor(cur_features.input_ids_org))
        all_input_mask.append(torch.tensor(cur_features.input_mask))
        all_segment_ids.append(torch.tensor(cur_features.segment_ids))
        all_lm_labels_ids.append(torch.tensor(cur_features.lm_label_ids))
        all_is_next.append(torch.tensor(0))
        all_tail_idxs.append(torch.tensor(cur_features.tail_idxs))
        if i in ids_neg:
            all_id_domain.append(torch.tensor(0))
        elif i in ids_pos:
            all_id_domain.append(torch.tensor(1))


    cur_tensors = (torch.stack(all_input_ids),
                   torch.stack(all_input_ids_org),
                   torch.stack(all_input_mask),
                   torch.stack(all_segment_ids),
                   torch.stack(all_lm_labels_ids),
                   torch.stack(all_is_next),
                   torch.stack(all_tail_idxs),
                   torch.stack(all_id_domain))

    return cur_tensors


def AugmentationData_Task(top_k, tokenizer, max_seq_length, add_org=None):
    top_k_shape = top_k["indices"].shape
    sentence_ids = top_k["indices"]

    all_input_ids = list()
    all_input_ids_org = list()
    all_input_mask = list()
    all_segment_ids = list()
    all_lm_labels_ids = list()
    all_is_next = list()
    all_tail_idxs = list()
    all_sentence_labels = list()
    all_sentiment_labels = list()

    add_org = tuple(t.to('cpu') for t in add_org)
    #input_ids_, input_ids_org_, input_mask_, segment_ids_, lm_label_ids_, is_next_, tail_idxs_, sentence_label_ = add_org
    input_ids_, input_ids_org_, input_mask_, segment_ids_, lm_label_ids_, is_next_, tail_idxs_, sentence_label_, sentiment_label_ = add_org

    ###
    #print("input_ids_",input_ids_.shape)
    #print("---")
    #print("sentence_ids",sentence_ids.shape)
    #print("---")
    #print("sentence_label_",sentence_label_.shape)
    #exit()


    for id_1, sent in enumerate(sentence_ids):
        for id_2, sent_id in enumerate(sent):

            t1 = data[str(int(sent_id))]['sentence']

            tokens_a = tokenizer.tokenize(t1)

            # tokenize
            cur_example = InputExample(guid=id, tokens_a=tokens_a, tokens_b=None, is_next=0)

            # transform sample to features
            cur_features = convert_example_to_features(cur_example, max_seq_length, tokenizer)

            all_input_ids.append(torch.tensor(cur_features.input_ids))
            all_input_ids_org.append(torch.tensor(cur_features.input_ids_org))
            all_input_mask.append(torch.tensor(cur_features.input_mask))
            all_segment_ids.append(torch.tensor(cur_features.segment_ids))
            all_lm_labels_ids.append(torch.tensor(cur_features.lm_label_ids))
            all_is_next.append(torch.tensor(0))
            all_tail_idxs.append(torch.tensor(cur_features.tail_idxs))
            all_sentence_labels.append(torch.tensor(sentence_label_[id_1]))
            all_sentiment_labels.append(torch.tensor(sentiment_label_[id_1]))

        all_input_ids.append(input_ids_[id_1])
        all_input_ids_org.append(input_ids_org_[id_1])
        all_input_mask.append(input_mask_[id_1])
        all_segment_ids.append(segment_ids_[id_1])
        all_lm_labels_ids.append(lm_label_ids_[id_1])
        all_is_next.append(is_next_[id_1])
        all_tail_idxs.append(tail_idxs_[id_1])
        all_sentence_labels.append(sentence_label_[id_1])
        all_sentiment_labels.append(sentiment_label_[id_1])


    cur_tensors = (torch.stack(all_input_ids),
                   torch.stack(all_input_ids_org),
                   torch.stack(all_input_mask),
                   torch.stack(all_segment_ids),
                   torch.stack(all_lm_labels_ids),
                   torch.stack(all_is_next),
                   torch.stack(all_tail_idxs),
                   torch.stack(all_sentence_labels),
                   torch.stack(all_sentiment_labels)
                   )


    return cur_tensors





def AugmentationData_Task_pos_and_neg_DT(top_k=None, tokenizer=None, max_seq_length=None, add_org=None, in_task_rep=None, in_domain_rep=None):
    '''
    top_k_shape = top_k.indices.shape
    sentence_ids = top_k.indices
    '''
    #top_k_shape = top_k["indices"].shape
    #sentence_ids = top_k["indices"]


    #input_ids_, input_ids_org_, input_mask_, segment_ids_, lm_label_ids_, is_next_, tail_idxs_, sentence_label_ = add_org
    input_ids_, input_ids_org_, input_mask_, segment_ids_, lm_label_ids_, is_next_, tail_idxs_, sentence_label_, sentiment_label_ = add_org


    #uniqe_type_id = torch.LongTensor(list(set(sentence_label_.tolist())))

    all_sentence_binary_label = list()
    #all_in_task_rep_comb = list()
    all_in_rep_comb = list()

    for id_1, num in enumerate(sentence_label_):
        #print([sentence_label_==num])
        #print(type([sentence_label_==num]))
        sentence_label_int = (sentence_label_==num).to(torch.long)
        #print(sentence_label_int)
        #print(sentence_label_int.shape)
        #print(in_task_rep[id_1].shape)
        #print(in_task_rep.shape)
        #exit()
        in_task_rep_append = in_task_rep[id_1].unsqueeze(0).expand(in_task_rep.shape[0],-1)
        in_domain_rep_append = in_domain_rep[id_1].unsqueeze(0).expand(in_domain_rep.shape[0],-1)
        #print(in_task_rep_append)
        #print(in_task_rep_append.shape)
        in_task_rep_comb = torch.cat((in_task_rep_append,in_task_rep),-1)
        in_domain_rep_comb = torch.cat((in_domain_rep_append,in_domain_rep),-1)
        #print(in_task_rep_comb)
        #print(in_task_rep_comb.shape)
        #exit()
        #sentence_label_int = sentence_label_int.to(torch.float32)
        #print(sentence_label_int)
        #exit()
        #all_sentence_binary_label.append(torch.tensor([1 if sentence_label_[id_1]==iid else 0 for iid in sentence_label_]))
        #all_sentence_binary_label.append(torch.tensor([1 if num==iid else 0 for iid in sentence_label_]))
        #print(in_task_rep_comb.shape)
        #print(in_domain_rep_comb.shape)
        in_rep_comb = torch.cat([in_domain_rep_comb,in_task_rep_comb],-1)
        #print(in_rep.shape)
        #exit()
        all_sentence_binary_label.append(sentence_label_int)
        #all_in_task_rep_comb.append(in_task_rep_comb)
        all_in_rep_comb.append(in_rep_comb)
    all_sentence_binary_label = torch.stack(all_sentence_binary_label)
    #all_in_task_rep_comb = torch.stack(all_in_task_rep_comb)
    all_in_rep_comb = torch.stack(all_in_rep_comb)

    #cur_tensors = (all_in_task_rep_comb, all_sentence_binary_label)
    cur_tensors = (all_in_rep_comb, all_sentence_binary_label)

    return cur_tensors




def AugmentationData_Task_pos_and_neg(top_k=None, tokenizer=None, max_seq_length=None, add_org=None, in_task_rep=None):
    '''
    top_k_shape = top_k.indices.shape
    sentence_ids = top_k.indices
    '''
    #top_k_shape = top_k["indices"].shape
    #sentence_ids = top_k["indices"]


    #input_ids_, input_ids_org_, input_mask_, segment_ids_, lm_label_ids_, is_next_, tail_idxs_, sentence_label_ = add_org
    input_ids_, input_ids_org_, input_mask_, segment_ids_, lm_label_ids_, is_next_, tail_idxs_, sentence_label_, sentiment_label_ = add_org


    #uniqe_type_id = torch.LongTensor(list(set(sentence_label_.tolist())))

    all_sentence_binary_label = list()
    all_in_task_rep_comb = list()

    for id_1, num in enumerate(sentence_label_):
        #print([sentence_label_==num])
        #print(type([sentence_label_==num]))
        sentence_label_int = (sentence_label_==num).to(torch.long)
        #print(sentence_label_int)
        #print(sentence_label_int.shape)
        #print(in_task_rep[id_1].shape)
        #print(in_task_rep.shape)
        #exit()
        in_task_rep_append = in_task_rep[id_1].unsqueeze(0).expand(in_task_rep.shape[0],-1)
        #print(in_task_rep_append)
        #print(in_task_rep_append.shape)
        in_task_rep_comb = torch.cat((in_task_rep_append,in_task_rep),-1)
        #print(in_task_rep_comb)
        #print(in_task_rep_comb.shape)
        #exit()
        #sentence_label_int = sentence_label_int.to(torch.float32)
        #print(sentence_label_int)
        #exit()
        #all_sentence_binary_label.append(torch.tensor([1 if sentence_label_[id_1]==iid else 0 for iid in sentence_label_]))
        #all_sentence_binary_label.append(torch.tensor([1 if num==iid else 0 for iid in sentence_label_]))
        all_sentence_binary_label.append(sentence_label_int)
        all_in_task_rep_comb.append(in_task_rep_comb)
    all_sentence_binary_label = torch.stack(all_sentence_binary_label)
    all_in_task_rep_comb = torch.stack(all_in_task_rep_comb)

    cur_tensors = (all_in_task_rep_comb, all_sentence_binary_label)

    return cur_tensors




class Dataset_noNext(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):

        self.vocab_size = tokenizer.vocab_size
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = [] # map sample index to doc and line

        # load samples into memory
        if on_memory:
            self.all_docs = []
            doc = []
            self.corpus_lines = 0
            with open(corpus_path, "r", encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    line = line.strip()
                    if line == "":
                        self.all_docs.append(doc)
                        doc = []
                        #remove last added sample because there won't be a subsequent line anymore in the doc
                        self.sample_to_doc.pop()
                    else:
                        #store as one sample
                        sample = {"doc_id": len(self.all_docs),
                                  "line": len(doc)}
                        self.sample_to_doc.append(sample)
                        doc.append(line)
                        self.corpus_lines = self.corpus_lines + 1

            # if last row in file is not empty
            if self.all_docs[-1] != doc:
                self.all_docs.append(doc)
                self.sample_to_doc.pop()

            self.num_docs = len(self.all_docs)

        # load samples later lazily from disk
        else:
            if self.corpus_lines is None:
                with open(corpus_path, "r", encoding=encoding) as f:
                    self.corpus_lines = 0
                    for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        if line.strip() == "":
                            self.num_docs += 1
                        else:
                            self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() != "":
                        self.num_docs += 1

            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.corpus_lines - self.num_docs - 1

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)

        #t1, t2, is_next_label = self.random_sent(item)
        t1, is_next_label = self.random_sent(item)
        if is_next_label == None:
            is_next_label = 0


        #tokens_a = self.tokenizer.tokenize(t1)
        tokens_a = tokenizer.tokenize(t1)
        '''
        if "</s>" in tokens_a:
            print("Have more than 1 </s>")
            #tokens_a[tokens_a.index("<s>")] = "s"
            for i in range(len(tokens_a)):
                if tokens_a[i] == "</s>":
                    tokens_a[i] = "s"
        '''
        #tokens_b = self.tokenizer.tokenize(t2)

        # tokenize
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=None, is_next=is_next_label)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_ids_org),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.is_next),
                       torch.tensor(cur_features.tail_idxs))

        return cur_tensors

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)
        return t1, None

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            return t1, t2
            #return t1
        else:
            if self.line_buffer is None:
                # read first non-empty line of file
                while t1 == "" :
                    t1 = next(self.file).strip()
            else:
                # use t2 from previous iteration as new t1
                t1 = self.line_buffer
                # skip empty rows that are used for separating documents and keep track of current doc id
                while t1 == "":
                    t1 = next(self.file).strip()
                    self.current_doc = self.current_doc+1
            self.line_buffer = next(self.file).strip()

        assert t1 != ""
        return t1, t2


    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        for _ in range(10):
            if self.on_memory:
                rand_doc_idx = random.randint(0, len(self.all_docs)-1)
                rand_doc = self.all_docs[rand_doc_idx]
                line = rand_doc[random.randrange(len(rand_doc))]
            else:
                rand_index = random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)
                #pick random line
                for _ in range(rand_index):
                    line = self.get_next_line()
            #check if our picked random line is really from another doc like we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line

    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = next(self.random_file).strip()
            #keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_ids_org, input_mask, segment_ids, is_next, lm_label_ids, tail_idxs):
        self.input_ids = input_ids
        self.input_ids_org = input_ids_org
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.tail_idxs = tail_idxs


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):

        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15
            #candidate_id = random.randint(0,tokenizer.vocab_size)
            #print(tokenizer.convert_ids_to_tokens(candidate_id))


            # 80% randomly change token to mask token
            if prob < 0.8:
                #tokens[i] = "[MASK]"
                tokens[i] = "<mask>"

            # 10% randomly change token to random token
            elif prob < 0.9:
                #tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]
                #tokens[i] = tokenizer.convert_ids_to_tokens(candidate_id)
                candidate_id = random.randint(0,tokenizer.vocab_size)
                w = tokenizer.convert_ids_to_tokens(candidate_id)
                '''
                if tokens[i] == None:
                    candidate_id = 100
                    w = tokenizer.convert_ids_to_tokens(candidate_id)
                '''
                tokens[i] = w


            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                #output_label.append(tokenizer.vocab[token])
                w = tokenizer.convert_tokens_to_ids(token)
                if w!= None:
                    output_label.append(w)
                else:
                    print("Have no this tokens in ids")
                    exit()
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                #output_label.append(tokenizer.vocab["<unk>"])
                w = tokenizer.convert_tokens_to_ids("<unk>")
                output_label.append(w)
                logger.warning("Cannot find token '{}' in vocab. Using <unk> insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    #now tokens_a is input_ids
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 2)

    #print(tokens_a)
    tokens_a_org = tokens_a.copy()
    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    #print("----")
    #print(tokens_a)
    #print(tokens_a_org)
    #exit()
    #print(t1_label)
    #exit()
    #tokens_b, t2_label = random_word(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP
    #lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
    lm_label_ids = ([-1] + t1_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    tokens_org = []
    segment_ids = []
    tokens.append("<s>")
    tokens_org.append("<s>")
    segment_ids.append(0)
    for i, token in enumerate(tokens_a):
        if token!="</s>":
            tokens.append(tokens_a[i])
            tokens_org.append(tokens_a_org[i])
            segment_ids.append(0)
        else:
            tokens.append("s")
            tokens_org.append("s")
            segment_ids.append(0)
    tokens.append("</s>")
    tokens_org.append("</s>")
    segment_ids.append(0)

    #tokens.append("[SEP]")
    #segment_ids.append(1)

    #input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.encode(tokens, add_special_tokens=False)
    input_ids_org = tokenizer.encode(tokens_org, add_special_tokens=False)
    tail_idxs = len(input_ids)-1

    #print(input_ids)
    input_ids = [w if w!=None else 0 for w in input_ids]
    input_ids_org = [w if w!=None else 0 for w in input_ids_org]
    #print(input_ids)
    #exit()

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    pad_id = tokenizer.convert_tokens_to_ids("<pad>")
    while len(input_ids) < max_seq_length:
        input_ids.append(pad_id)
        input_ids_org.append(pad_id)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    try:
        assert len(input_ids) == max_seq_length
        assert len(input_ids_org) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
    except:
        print("!!!Warning!!!")
        input_ids = input_ids[:max_seq_length-1]
        if 2 not in input_ids:
            input_ids += [2]
        else:
            input_ids += [pad_id]
        input_ids_org = input_ids_org[:max_seq_length-1]
        if 2 not in input_ids_org:
            input_ids_org += [2]
        else:
            input_ids_org += [pad_id]
        input_mask = input_mask[:max_seq_length-1]+[0]
        segment_ids = segment_ids[:max_seq_length-1]+[0]
        lm_label_ids = lm_label_ids[:max_seq_length-1]+[-1]
    '''
    flag=False
    if len(input_ids) != max_seq_length:
        print(len(input_ids))
        flag=True
    if len(input_ids_org) != max_seq_length:
        print(len(input_ids_org))
        flag=True
    if len(input_mask) != max_seq_length:
        print(len(input_mask))
        flag=True
    if len(segment_ids) != max_seq_length:
        print(len(segment_ids))
        flag=True
    if len(lm_label_ids) != max_seq_length:
        print(len(lm_label_ids))
        flag=True
    if flag == True:
        print("1165")
        exit()
    '''

    '''
    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("LM label: %s " % (lm_label_ids))
        logger.info("Is next sentence label: %s " % (example.is_next))
    '''

    features = InputFeatures(input_ids=input_ids,
                             input_ids_org = input_ids_org,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next,
                             tail_idxs=tail_idxs)
    return features


def main():
    parser = argparse.ArgumentParser()

    parser = get_parameter(parser)

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model, do_lower_case=args.do_lower_case)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model)


    #train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        print("Loading Train Dataset", args.data_dir_indomain)
        #train_dataset = Dataset_noNext(args.data_dir, tokenizer, seq_len=args.max_seq_length, corpus_lines=None, on_memory=args.on_memory)
        all_type_sentence, train_dataset = in_Domain_Task_Data_mutiple(args.data_dir_indomain, tokenizer, args.max_seq_length)
        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()



    # Prepare model
    model = RobertaForMaskedLMDomainTask.from_pretrained(args.pretrain_model, output_hidden_states=True, return_dict=True, num_labels=args.num_labels_task)
    #model = RobertaForSequenceClassification.from_pretrained(args.pretrain_model, output_hidden_states=True, return_dict=True, num_labels=args.num_labels_task)
    model.to(device)



    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        '''
        for par in param_optimizer:
            print(par[0])
        exit()
        '''
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_train_optimization_steps*0.1), num_training_steps=num_train_optimization_steps)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
                exit()

            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)


        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)



    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            #all_type_sentence_sampler = RandomSampler(all_type_sentence)
        else:
            #TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
            #all_type_sentence_sampler = DistributedSampler(all_type_sentence)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        #all_type_sentence_dataloader = DataLoader(all_type_sentence, sampler=all_type_sentence_sampler, batch_size=len(all_type_sentence_label))

        output_loss_file = os.path.join(args.output_dir, "loss")
        loss_fout = open(output_loss_file, 'w')


        output_loss_file_no_pseudo = os.path.join(args.output_dir, "loss_no_pseudo")
        loss_fout_no_pseudo = open(output_loss_file_no_pseudo, 'w')
        model.train()




        #alpha = float(1/(args.num_train_epochs*len(train_dataloader)))
        #alpha = float(1/args.num_train_epochs)
        alpha = float(1)
        #k=8
        #k=16
        #k = args.K
        k = 10
        #k = 2
        #retrive_gate = args.num_labels_task
        #retrive_gate = len(train_dataset)/100
        retrive_gate = 1
        all_type_sentence_label = list()
        all_previous_sentence_label = list()
        all_type_sentiment_label = list()
        all_previous_sentiment_label = list()
        top_k_all_type = dict()
        bottom_k_all_type = dict()
        for epo in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch_ in enumerate(tqdm(train_dataloader, desc="Iteration")):


                #######################
                ######################
                ###Init 8 type sentence
                ###Init 2 type sentiment
                if (step == 0) and (epo == 0):
                    #batch_ = tuple(t.to(device) for t in batch_)
                    #all_type_sentence_ = tuple(t.to(device) for t in all_type_sentence)

                    input_ids_ = torch.stack([line[0] for line in all_type_sentence]).to(device)
                    input_ids_org_ = torch.stack([line[1] for line in all_type_sentence]).to(device)
                    input_mask_ = torch.stack([line[2] for line in all_type_sentence]).to(device)
                    segment_ids_ = torch.stack([line[3] for line in all_type_sentence]).to(device)
                    lm_label_ids_ = torch.stack([line[4] for line in all_type_sentence]).to(device)
                    is_next_ = torch.stack([line[5] for line in all_type_sentence]).to(device)
                    tail_idxs_ = torch.stack([line[6] for line in all_type_sentence]).to(device)
                    sentence_label_ = torch.stack([line[7] for line in all_type_sentence]).to(device)
                    sentiment_label_ = torch.stack([line[8] for line in all_type_sentence]).to(device)

                    with torch.no_grad():

                        #in_domain_rep_mean, in_task_rep_mean = model(input_ids_org=input_ids_org_, tail_idxs=tail_idxs_, attention_mask=input_mask_, func="in_domain_task_rep_mean")
                        in_domain_rep, in_task_rep = model(input_ids_org=input_ids_org_, tail_idxs=tail_idxs_, attention_mask=input_mask_, func="in_domain_task_rep")
                        # Search id from Docs and ranking via (Domain/Task)
                        #query_domain = in_domain_rep_mean.float().to("cpu")
                        query_domain = in_domain_rep.float().to("cpu")
                        query_domain = query_domain.unsqueeze(1)
                        #query_task = in_task_rep_mean.float().to("cpu")
                        query_task = in_task_rep.float().to("cpu")
                        query_task = query_task.unsqueeze(1)
                        query_domain_task = torch.cat([query_domain,query_task],2)


                        task_binary_classifier_weight, task_binary_classifier_bias = model(func="return_task_binary_classifier")
                        task_binary_classifier_weight = task_binary_classifier_weight[:int(task_binary_classifier_weight.shape[0]/n_gpu)][:]
                        task_binary_classifier_bias = task_binary_classifier_bias[:int(task_binary_classifier_bias.shape[0]/n_gpu)][:]
                        task_binary_classifier = return_Classifier(task_binary_classifier_weight, task_binary_classifier_bias, 768*2, 2)


                        domain_binary_classifier_weight, domain_binary_classifier_bias = model(func="return_domain_binary_classifier")
                        domain_binary_classifier_weight = domain_binary_classifier_weight[:int(domain_binary_classifier_weight.shape[0]/n_gpu)][:]
                        domain_binary_classifier_bias = domain_binary_classifier_bias[:int(domain_binary_classifier_bias.shape[0]/n_gpu)][:]
                        domain_binary_classifier = return_Classifier(domain_binary_classifier_weight, domain_binary_classifier_bias, 768*2, 2)


                        domain_task_binary_classifier_weight, domain_task_binary_classifier_bias = model(func="return_domain_task_binary_classifier")
                        domain_task_binary_classifier_weight = domain_task_binary_classifier_weight[:int(domain_task_binary_classifier_weight.shape[0]/n_gpu)][:]
                        domain_task_binary_classifier_bias = domain_task_binary_classifier_bias[:int(domain_task_binary_classifier_bias.shape[0]/n_gpu)][:]
                        domain_task_binary_classifier = return_Classifier(domain_task_binary_classifier_weight, domain_task_binary_classifier_bias, 768*4, 2)

                        #start = time.time()
                        query_domain = query_domain.expand(-1, docs_tail.shape[0], -1)
                        query_task = query_task.expand(-1, docs_head.shape[0], -1)
                        query_domain_task = query_domain_task.expand(-1, docs_head.shape[0], -1)

                        #################
                        #################
                        #Ranking

                        #LeakyReLU = torch.nn.LeakyReLU()
                        #Domain logit
                        '''
                        domain_binary_logit = LeakyReLU(domain_binary_classifier(docs_tail))
                        domain_binary_logit = domain_binary_logit[:,:,1] - domain_binary_logit[:,:,0]
                        domain_binary_logit = domain_binary_logit.squeeze(1).unsqueeze(0).expand(sentiment_label_.shape[0], -1)
                        domain_binary_logit = domain_binary_classifier(torch.cat([query_domain, docs_tail[:,0,:].unsqueeze(0).expand(sentiment_label_.shape[0], -1, -1)], dim=2))
                        target = torch.zeros(domain_binary_logit.shape[0], domain_binary_logit.shape[1], dtype=torch.long)
                        #domain_binary_logit = domain_binary_logit[:,:,1] - domain_binary_logit[:,:,0]
                        domain_binary_logit = ce_loss(domain_binary_logit.view(-1, 2), target.view(-1)).reshape(domain_binary_logit.shape[0],domain_binary_logit.shape[1])
                        '''

                        #Task logit
                        '''
                        task_binary_logit = task_binary_classifier(torch.cat([query_task, docs_head[:,0,:].unsqueeze(0).expand(sentiment_label_.shape[0], -1, -1)], dim=2))
                        #task_binary_logit = task_binary_logit[:,:,1] - task_binary_logit[:,:,0]
                        #target = torch.zeros(task_binary_logit.shape[0], task_binary_logit.shape[1], dtype=torch.long)
                        task_binary_logit = ce_loss(task_binary_logit.view(-1, 2), target.view(-1)).reshape(task_binary_logit.shape[0],task_binary_logit.shape[1])
                        '''

                        #Domain Task logit
                        #print(query_domain_task.shape)
                        #print(docs_head.shape)
                        domain_task_binary_logit = domain_task_binary_classifier(torch.cat([query_domain_task, docs_tail_head[:,0,:].unsqueeze(0).expand(sentiment_label_.shape[0], -1, -1)], dim=2))
                        #domain_task_binary_logit = domain_task_binary_logit[:,:,1] - domain_task_binary_logit[:,:,0]
                        target = torch.zeros(domain_task_binary_logit.shape[0], domain_task_binary_logit.shape[1], dtype=torch.long)
                        domain_task_binary_logit = ce_loss(domain_task_binary_logit.view(-1, 2), target.view(-1)).reshape(domain_task_binary_logit.shape[0],domain_task_binary_logit.shape[1])

                        '''
                        #end = time.time()
                        #print("Time:", (end-start)/60)
                        ######
                        #[batch_size, 36603]
                        #results_all_type = domain_binary_logit + task_binary_logit
                        #del domain_binary_logit, task_binary_logit
                        #bottom_k_all_type = torch.topk(results_all_type, k, dim=1, largest=False, sorted=False)
                        #bottom_k_all_type = torch.topk(results_all_type, k, dim=1, largest=False, sorted=False)
                        domain_top_k_all_type = torch.topk(domain_binary_logit, k, dim=1, largest=True, sorted=False)
                        #domain_bottom_k_all_type = torch.stack(random.choices(domain_binary_logit[:,k:], k=k))
                        perm = torch.randperm(domain_binary_logit.shape[1])
                        domain_bottom_k_all_type_indices = perm[:k]
                        domain_bottom_k_all_type_values = domain_binary_logit[:,domain_bottom_k_all_type_indices]
                        domain_bottom_k_all_type_indices = torch.stack(args.train_batch_size*[domain_bottom_k_all_type_indices])


                        #top_k_all_type = torch.topk(results_all_type, k, dim=1, largest=True, sorted=False)
                        task_top_k_all_type = torch.topk(task_binary_logit, k, dim=1, largest=True, sorted=False)
                        '''

                        domain_task_top_k_all_type = torch.topk(domain_task_binary_logit, k, dim=1, largest=True, sorted=False)

                        #del domain_task_binary_logit, domain_binary_logit, task_binary_logit
                        del domain_task_binary_logit

                        all_type_sentiment_label = sentiment_label_.to('cpu')


                        #domain_bottom_k_all_type = {"values":domain_bottom_k_all_type_values, "indices":domain_bottom_k_all_type_indices}
                        #domain_top_k_all_type = {"values":domain_top_k_all_type.values, "indices":domain_top_k_all_type.indices}
                        #task_top_k_all_type = {"values":task_top_k_all_type.values, "indices":task_top_k_all_type.indices}
                        domain_task_top_k_all_type = {"values":domain_task_top_k_all_type.values, "indices":domain_task_top_k_all_type.indices}

                ######################
                ######################


                ###Normal mode
                batch_ = tuple(t.to(device) for t in batch_)
                input_ids_, input_ids_org_, input_mask_, segment_ids_, lm_label_ids_, is_next_, tail_idxs_, sentence_label_, sentiment_label_ = batch_


                ###
                # Generate query representation
                in_domain_rep, in_task_rep = model(input_ids_org=input_ids_org_, tail_idxs=tail_idxs_, attention_mask=input_mask_, func="in_domain_task_rep")


                #if (step%10 == 0) or (sentence_label_.shape[0] != args.train_batch_size):
                if (step%retrive_gate == 0) or (sentiment_label_.shape[0] != args.train_batch_size):
                    with torch.no_grad():
                        query_domain = in_domain_rep.float().to("cpu")
                        query_domain = query_domain.unsqueeze(1)
                        #query_task = in_task_rep_mean.float().to("cpu")
                        query_task = in_task_rep.float().to("cpu")
                        query_task = query_task.unsqueeze(1)
                        query_domain_task = torch.cat([query_domain,query_task],2)


                        task_binary_classifier_weight, task_binary_classifier_bias = model(func="return_task_binary_classifier")
                        task_binary_classifier_weight = task_binary_classifier_weight[:int(task_binary_classifier_weight.shape[0]/n_gpu)][:]
                        task_binary_classifier_bias = task_binary_classifier_bias[:int(task_binary_classifier_bias.shape[0]/n_gpu)][:]
                        task_binary_classifier = return_Classifier(task_binary_classifier_weight, task_binary_classifier_bias, 768*2, 2)


                        domain_binary_classifier_weight, domain_binary_classifier_bias = model(func="return_domain_binary_classifier")
                        domain_binary_classifier_weight = domain_binary_classifier_weight[:int(domain_binary_classifier_weight.shape[0]/n_gpu)][:]
                        domain_binary_classifier_bias = domain_binary_classifier_bias[:int(domain_binary_classifier_bias.shape[0]/n_gpu)][:]
                        domain_binary_classifier = return_Classifier(domain_binary_classifier_weight, domain_binary_classifier_bias, 768*2, 2)


                        domain_task_binary_classifier_weight, domain_task_binary_classifier_bias = model(func="return_domain_task_binary_classifier")
                        domain_task_binary_classifier_weight = domain_task_binary_classifier_weight[:int(domain_task_binary_classifier_weight.shape[0]/n_gpu)][:]
                        domain_task_binary_classifier_bias = domain_task_binary_classifier_bias[:int(domain_task_binary_classifier_bias.shape[0]/n_gpu)][:]
                        domain_task_binary_classifier = return_Classifier(domain_task_binary_classifier_weight, domain_task_binary_classifier_bias, 768*4, 2)

                        #start = time.time()
                        #query_domain = query_domain.expand(-1, docs.shape[0], -1)
                        query_domain = query_domain.expand(-1, docs_tail.shape[0], -1)
                        #query_task = query_task.expand(-1, docs.shape[0], -1)
                        query_task = query_task.expand(-1, docs_head.shape[0], -1)
                        #print(docs_head.shape)
                        #print(query_domain_task.shape)
                        #exit()
                        query_domain_task = query_domain_task.expand(-1, docs_head.shape[0], -1)

                        #################
                        #################
                        #Ranking

                        #LeakyReLU = torch.nn.LeakyReLU()
                        #Domain logit
                        '''
                        domain_binary_logit = LeakyReLU(domain_binary_classifier(docs_tail))
                        domain_binary_logit = domain_binary_logit[:,:,1] - domain_binary_logit[:,:,0]
                        domain_binary_logit = domain_binary_logit.squeeze(1).unsqueeze(0).expand(sentiment_label_.shape[0], -1)
                        '''

                        '''
                        domain_binary_logit = domain_binary_classifier(torch.cat([query_domain, docs_tail[:,0,:].unsqueeze(0).expand(sentiment_label_.shape[0], -1, -1)], dim=2))
                        target = torch.zeros(domain_binary_logit.shape[0], domain_binary_logit.shape[1], dtype=torch.long)
                        #domain_binary_logit = domain_binary_logit[:,:,1] - domain_binary_logit[:,:,0]
                        domain_binary_logit = ce_loss(domain_binary_logit.view(-1, 2), target.view(-1)).reshape(domain_binary_logit.shape[0],domain_binary_logit.shape[1])

                        #Task logit
                        task_binary_logit = task_binary_classifier(torch.cat([query_task, docs_head[:,0,:].unsqueeze(0).expand(sentiment_label_.shape[0], -1, -1)], dim=2))
                        #task_binary_logit = task_binary_logit[:,:,1] - task_binary_logit[:,:,0]
                        task_binary_logit = ce_loss(task_binary_logit.view(-1, 2), target.view(-1)).reshape(task_binary_logit.shape[0],task_binary_logit.shape[1])
                        '''

                        #Domain Task logit
                        #print(query_domain_task.shape)
                        #print(docs_head.shape)
                        domain_task_binary_logit = domain_task_binary_classifier(torch.cat([query_domain_task, docs_tail_head[:,0,:].unsqueeze(0).expand(sentiment_label_.shape[0], -1, -1)], dim=2))
                        #print(domain_task_binary_logit.shape)
                        #domain_task_binary_logit = domain_task_binary_logit[:,:,1] - domain_task_binary_logit[:,:,0]
                        target = torch.zeros(domain_task_binary_logit.shape[0], domain_task_binary_logit.shape[1], dtype=torch.long)
                        domain_task_binary_logit = ce_loss(domain_task_binary_logit.view(-1, 2), target.view(-1)).reshape(domain_task_binary_logit.shape[0],domain_task_binary_logit.shape[1])


                        #end = time.time()
                        #print("Time:", (end-start)/60)
                        ######
                        '''
                        #[batch_size, 36603]
                        #results_all_type = domain_binary_logit + task_binary_logit
                        #del domain_binary_logit, task_binary_logit
                        #bottom_k_all_type = torch.topk(results_all_type, k, dim=1, largest=False, sorted=False)
                        #bottom_k_all_type = torch.topk(results_all_type, k, dim=1, largest=False, sorted=False)
                        domain_top_k = torch.topk(domain_binary_logit, k, dim=1, largest=True, sorted=False)
                        #domain_bottom_k = torch.stack(random.choices(domain_binary_logit[:,k:], k=k))
                        perm = torch.randperm(domain_binary_logit.shape[1])
                        domain_bottom_k_indices = perm[:k]
                        domain_bottom_k_values = domain_binary_logit[:,domain_bottom_k_indices]
                        domain_bottom_k_indices = torch.stack(args.train_batch_size*[domain_bottom_k_indices])


                        #top_k_all_type = torch.topk(results_all_type, k, dim=1, largest=True, sorted=False)
                        task_top_k = torch.topk(task_binary_logit, k, dim=1, largest=True, sorted=False)
                        '''


                        domain_task_top_k = torch.topk(domain_task_binary_logit, k, dim=1, largest=True, sorted=False)


                        #del domain_task_binary_logit, domain_binary_logit, task_binary_logit
                        del domain_task_binary_logit

                        all_previous_sentiment_label = sentiment_label_.to('cpu')

                        ######
                        #results = domain_binary_logit + task_binary_logit
                        #del domain_binary_logit, task_binary_logit
                        #bottom_k = torch.topk(results, k, dim=1, largest=False, sorted=False)
                        #bottom_k = {"values":bottom_k.values, "indices":bottom_k.indices}
                        #top_k = torch.topk(results, k, dim=1, largest=True, sorted=False)
                        #top_k = {"values":top_k.values, "indices":top_k.indices}
                        #del results


                        #domain_bottom_k = {"values":domain_bottom_k_values, "indices":domain_bottom_k_indices}
                        #domain_top_k = {"values":domain_top_k.values, "indices":domain_top_k.indices}
                        #task_top_k = {"values":task_top_k.values, "indices":task_top_k.indices}
                        domain_task_top_k = {"values":domain_task_top_k.values, "indices":domain_task_top_k.indices}



                        #bottom_k_previous = {"values":torch.cat((bottom_k["values"], bottom_k_all_type["values"]),0), "indices":torch.cat((bottom_k["indices"], bottom_k_all_type["indices"]),0)}
                        #top_k_previous = {"values":torch.cat((top_k["values"], top_k_all_type["values"]),0), "indices":torch.cat((top_k["indices"], top_k_all_type["indices"]),0)}
                        #all_previous_sentiment_label = torch.cat((all_previous_sentiment_label, all_type_sentiment_label))

                        #domain_bottom_k_previous = {"values":torch.cat((domain_bottom_k["values"], domain_bottom_k_all_type["values"]),0), "indices":torch.cat((domain_bottom_k["indices"], domain_bottom_k_all_type["indices"]),0)}
                        #domain_top_k_previous = {"values":torch.cat((domain_top_k["values"], domain_top_k_all_type["values"]),0), "indices":torch.cat((domain_top_k["indices"], domain_top_k_all_type["indices"]),0)}
                        #task_top_k_previous = {"values":torch.cat((task_top_k["values"], task_top_k_all_type["values"]),0), "indices":torch.cat((task_top_k["indices"], task_top_k_all_type["indices"]),0)}
                        domain_task_top_k_previous = {"values":torch.cat((domain_task_top_k["values"], domain_task_top_k_all_type["values"]),0), "indices":torch.cat((domain_task_top_k["indices"], domain_task_top_k_all_type["indices"]),0)}

                        all_previous_sentiment_label = torch.cat((all_previous_sentiment_label, all_type_sentiment_label))
                else:
                    ###Need to fix --> choice
                    used_idx = torch.tensor([random.choice(((all_previous_sentiment_label==int(idx_)).nonzero()).tolist())[0] for idx_ in sentiment_label_])
                    #top_k = {"values":top_k_previous["values"].index_select(0,used_idx), "indices":top_k_previous["indices"].index_select(0,used_idx)}
                    #domain_top_k = {"values":domain_top_k_previous["values"].index_select(0,used_idx), "indices":domain_top_k_previous["indices"].index_select(0,used_idx)}
                    #task_top_k = {"values":task_top_k_previous["values"].index_select(0,used_idx), "indices":task-top_k_previous["indices"].index_select(0,used_idx)}
                    domain_task_top_k = {"values":domain_task_top_k_previous["values"].index_select(0,used_idx), "indices":domain_task_top_k_previous["indices"].index_select(0,used_idx)}

                    #bottom_k = {"values":bottom_k_previous["values"].index_select(0,used_idx), "indices":bottom_k_previous["indices"].index_select(0,used_idx)}
                    #domaion_bottom_k = {"values":domain_bottom_k_previous["values"].index_select(0,used_idx), "indices":domain_bottom_k_previous["indices"].index_select(0,used_idx)}


                #################
                #################
                '''
                #Train Domain Binary Classifier
                #batch = AugmentationData_Domain(bottom_k, tokenizer, args.max_seq_length)
                batch = AugmentationData_Domain(domain_top_k, domain_bottom_k, tokenizer, args.max_seq_length)
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_ids_org, input_mask, segment_ids, lm_label_ids, is_next, tail_idxs, domain_id = batch

                out_domain_rep_tail, out_domain_rep_head = model(input_ids_org=input_ids_org, lm_label=lm_label_ids, attention_mask=input_mask, func="in_domain_task_rep")

                ############Construct constrive instances
                comb_rep_pos = torch.cat([in_domain_rep,in_domain_rep.flip(0)], 1)
                in_domain_rep_ready = in_domain_rep.repeat(1,int(out_domain_rep_tail.shape[0]/in_domain_rep.shape[0])).reshape(out_domain_rep_head.shape[0],out_domain_rep_tail.shape[1])
                comb_rep_unknow = torch.cat([in_domain_rep_ready, out_domain_rep_head], 1)

                domain_binary_loss, domain_binary_logit = model(func="domain_binary_classifier", in_domain_rep=comb_rep_pos.to(device), out_domain_rep=comb_rep_unknow.to(device), domain_id=domain_id)
                ############


                #domain_binary_loss, domain_binary_logit, out_domain_rep_head, out_domain_rep_tail = model(input_ids_org=input_ids_org, lm_label=lm_label_ids, attention_mask=input_mask, func="domain_binary_classifier", in_domain_rep=in_domain_rep.to(device), domain_id=domain_id)

                #################
                #################
                ###Update_rep
                indices = domain_top_k["indices"].reshape(domain_top_k["indices"].shape[0]*domain_top_k["indices"].shape[1])
                indices_ = domain_bottom_k["indices"].reshape(domain_bottom_k["indices"].shape[0]*domain_bottom_k["indices"].shape[1])
                indices = torch.cat([indices,indices_],0)

                out_domain_rep_head = out_domain_rep_head.reshape(out_domain_rep_head.shape[0],1,out_domain_rep_head.shape[1]).to("cpu").data
                out_domain_rep_head.requires_grad=True

                out_domain_rep_tail = out_domain_rep_tail.reshape(out_domain_rep_tail.shape[0],1,out_domain_rep_tail.shape[1]).to("cpu").data
                out_domain_rep_tail.requires_grad=True


                with torch.no_grad():
                    #Exam here!!!
                    try:
                        docs_head.index_copy_(0, indices, out_domain_rep_head)
                        docs_tail.index_copy_(0, indices, out_domain_rep_tail)
                    except:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("head",out_domain_rep_head.shape)
                        print("tail",out_domain_rep_head.shape)
                        print("doc_h",docs_head.shape)
                        print("doc_t",docs_tail.shape)
                        print("ind",indices.shape)
                '''



                #################
                #################
                #Train Task Binary Classifier    in domain
                #Pseudo Task --> Won't bp to PLM: only train classifier [In domain data]
                batch = AugmentationData_Task_pos_and_neg_DT(top_k=None, tokenizer=tokenizer, max_seq_length=args.max_seq_length, add_org=batch_, in_task_rep=in_task_rep, in_domain_rep=in_domain_rep)
                batch = tuple(t.to(device) for t in batch)
                all_in_task_rep_comb, all_sentence_binary_label = batch
                task_binary_loss, task_binary_logit = model(all_in_task_rep_comb=all_in_task_rep_comb, all_sentence_binary_label=all_sentence_binary_label, func="domain_task_binary_classifier")


                #################
                #################
                #Train Task org - finetune
                #split into: in_dom and query_  --> different weight
                task_loss_org, class_logit_org = model(input_ids_org=input_ids_org_, sentence_label=sentiment_label_, attention_mask=input_mask_, func="task_class")


                #################
                #################
                #Task Level   including outdomain
                '''
                batch = AugmentationData_Task(task_top_k, tokenizer, args.max_seq_length, add_org=batch_)
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_ids_org, input_mask, segment_ids, lm_label_ids, is_next, tail_idxs, sentence_label, sentiment_label = batch
                out_domain_rep_tail, out_domain_rep_head = model(input_ids_org=input_ids_org, tail_idxs=tail_idxs, attention_mask=input_mask, func="in_domain_task_rep")
                ###
                batch = AugmentationData_Task_pos_and_neg(top_k=None, tokenizer=tokenizer, max_seq_length=args.max_seq_length, add_org=batch, in_task_rep=out_domain_rep_head)
                batch = tuple(t.to(device) for t in batch)
                all_in_task_rep_comb, all_sentence_binary_label = batch
                task_loss_query, task_binary_logit = model(all_in_task_rep_comb=all_in_task_rep_comb, all_sentence_binary_label=all_sentence_binary_label, func="task_binary_classifier")
                ###

                #################
                #################
                ###Update_rep
                indices = task_top_k["indices"].reshape(task_top_k["indices"].shape[0]*task_top_k["indices"].shape[1])

                out_domain_rep_head = out_domain_rep_head[input_ids_org_.shape[0]:,:]
                out_domain_rep_head = out_domain_rep_head.reshape(out_domain_rep_head.shape[0],1,out_domain_rep_head.shape[1]).to("cpu").data
                out_domain_rep_head.requires_grad=True

                out_domain_rep_tail = out_domain_rep_tail[input_ids_org_.shape[0]:,:]
                out_domain_rep_tail = out_domain_rep_tail.reshape(out_domain_rep_tail.shape[0],1,out_domain_rep_tail.shape[1]).to("cpu").data
                out_domain_rep_tail.requires_grad=True

                with torch.no_grad():
                    try:
                        docs_head.index_copy_(0, indices, out_domain_rep_head)
                        docs_tail.index_copy_(0, indices, out_domain_rep_tail)
                    except:
                        print("head",out_domain_rep_head.shape)
                        print("head",out_domain_rep_head.get_device())
                        print("tail",out_domain_rep_head.shape)
                        print("tail",out_domain_rep_head.get_device())
                        print("doc_h",docs_head.shape)
                        print("doc_h",docs_head.get_device())
                        print("doc_t",docs_tail.shape)
                        print("doc_t",docs_tail.get_device())
                        print("ind",indices.shape)
                        print("ind",indices.get_device())
                '''

                ##############################
                ##############################

                #################
                #################
                #Domain-Task Level
                batch = AugmentationData_Task(domain_task_top_k, tokenizer, args.max_seq_length, add_org=batch_)
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_ids_org, input_mask, segment_ids, lm_label_ids, is_next, tail_idxs, sentence_label, sentiment_label = batch
                out_domain_rep_tail, out_domain_rep_head = model(input_ids_org=input_ids_org, tail_idxs=tail_idxs, attention_mask=input_mask, func="in_domain_task_rep")
                ###
                batch = AugmentationData_Task_pos_and_neg_DT(top_k=None, tokenizer=tokenizer, max_seq_length=args.max_seq_length, add_org=batch, in_task_rep=out_domain_rep_head, in_domain_rep=out_domain_rep_tail)
                batch = tuple(t.to(device) for t in batch)
                all_in_task_rep_comb, all_sentence_binary_label = batch
                domain_task_loss_query, domain_task_binary_logit = model(all_in_task_rep_comb=all_in_task_rep_comb, all_sentence_binary_label=all_sentence_binary_label, func="domain_task_binary_classifier")
                ###


                #################
                #################
                ###Update_rep
                indices = domain_task_top_k["indices"].reshape(domain_task_top_k["indices"].shape[0]*domain_task_top_k["indices"].shape[1])

                out_domain_rep_head = out_domain_rep_head[input_ids_org_.shape[0]:,:]
                out_domain_rep_head = out_domain_rep_head.reshape(out_domain_rep_head.shape[0],1,out_domain_rep_head.shape[1]).to("cpu").data
                out_domain_rep_head.requires_grad=True

                out_domain_rep_tail = out_domain_rep_tail[input_ids_org_.shape[0]:,:]
                out_domain_rep_tail = out_domain_rep_tail.reshape(out_domain_rep_tail.shape[0],1,out_domain_rep_tail.shape[1]).to("cpu").data
                out_domain_rep_tail.requires_grad=True

                with torch.no_grad():
                    try:
                        docs_head.index_copy_(0, indices, out_domain_rep_head)
                        docs_tail.index_copy_(0, indices, out_domain_rep_tail)
                    except:
                        print("head",out_domain_rep_head.shape)
                        print("head",out_domain_rep_head.get_device())
                        print("tail",out_domain_rep_head.shape)
                        print("tail",out_domain_rep_head.get_device())
                        print("doc_h",docs_head.shape)
                        print("doc_h",docs_head.get_device())
                        print("doc_t",docs_tail.shape)
                        print("doc_t",docs_tail.get_device())
                        print("ind",indices.shape)
                        print("ind",indices.get_device())

                ##############################
                ##############################

                if n_gpu > 1:
                    #pseudo = (task_loss_query.mean()*alpha*epo)
                    #pseudo = (task_loss_query.mean()*alpha)
                    #pseudo = (task_loss_query.mean()*alpha*epo)+(masked_loss_query.mean()*alpha*epo)
                    #loss = domain_binary_loss.mean() + task_binary_loss.mean() + task_loss_org.mean() + pseudo + domain_task_loss_query.mean() #+ mlm_loss.mean()
                    #loss = task_binary_loss.mean() + task_loss_org.mean() + pseudo + domain_task_loss_query.mean() #+ mlm_loss.mean()
                    loss = task_loss_org.mean() + domain_task_loss_query.mean()
                    #loss = domain_binary_loss.mean() + task_binary_loss.mean() + task_loss_org.mean() + masked_loss_org.mean() + pseudo
                else:
                    #pseudo = (task_loss_query*alpha*epo)
                    #pseudo = (task_loss_query*alpha)
                    #pseudo = (task_loss_query*alpha*epo)+(masked_loss_query*alpha*epo)
                    #loss = domain_binary_loss + task_binary_loss + task_loss_org + pseudo + domain_task_loss_query #+ mlm_loss
                    #loss = task_binary_loss + task_loss_org + pseudo + domain_task_loss_query #+ mlm_loss
                    loss = task_loss_org + domain_task_loss_query #+ mlm_loss
                    #loss = domain_binary_loss + task_binary_loss + task_loss_org + masked_loss_org + pseudo


                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                ###
                loss_fout.write("{}\n".format(loss.item()))
                ###

                ###
                #loss_fout_no_pseudo.write("{}\n".format(loss.item()-pseudo.item()))
                ###

                tr_loss += loss.item()
                #nb_tr_examples += input_ids.size(0)
                nb_tr_examples += input_ids_.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        #lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    ###
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    ###

                    optimizer.step()
                    ###
                    scheduler.step()
                    ###
                    #optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1


            if epo < 2:
                continue
            else:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                #output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(global_step))
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(epo))
                torch.save(model_to_save.state_dict(), output_model_file)
            ####
            '''
            #if args.num_train_epochs/args.augment_times in [1,2,3]:
            if (args.num_train_epochs/(args.augment_times/5))%5 == 0:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(global_step))
                torch.save(model_to_save.state_dict(), output_model_file)
            '''
            ####

        loss_fout.close()

        # Save a trained model
        logger.info("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        if args.do_train:
            torch.save(model_to_save.state_dict(), output_model_file)



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        #total_length = len(tokens_a) + len(tokens_b)
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        else:
            tokens_a.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


if __name__ == "__main__":
    main()
