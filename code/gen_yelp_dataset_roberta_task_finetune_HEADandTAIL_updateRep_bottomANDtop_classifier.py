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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification
#from transformers.modeling_roberta import RobertaForMaskedLMDomainTask
from transformers.modeling_roberta_updateRep import RobertaForMaskedLMDomainTask
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


#def default_all_type_sentence(batch):



def return_Classifier(weight, bias, dim_in, dim_out):
    classifier = torch.nn.Linear(dim_in, dim_out , bias=True)
    classifier.weight.data = weight.to("cpu")
    classifier.bias.data = bias.to("cpu")
    classifier.requires_grad=False
    return classifier


def load_GeneralDomain(dir_data_out):
    ###Test
    if dir_data_out=="data/open_domain_preprocessed_roberta/":
        docs = torch.load(dir_data_out+"opendomain_CLS.pt")
        with open(dir_data_out+"opendomain.json") as file:
            data = json.load(file)
        print("train.json Done")
        print("===========")
        docs = docs.unsqueeze(1)
        return docs, data
    ###


    ###
    elif dir_data_out=="data/yelp/":
        print("===========")
        print("Load CLS.pt and train.json")
        print("-----------")
        docs = torch.load(dir_data_out+"train_CLS.pt")
        print("CLS.pt Done")
        print(docs.shape)
        print("-----------")
        with open(dir_data_out+"train.json") as file:
            data = json.load(file)
        print("train.json Done")
        print("===========")
        return docs, data
    ###


    ###
    elif dir_data_out=="data/yelp_finetune_noword_10000/":
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
        return docs_head, docs_tail, data
    ###

    ###
    elif dir_data_out=="data/opendomain_finetune_noword_10000/":
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
        return docs_head, docs_tail, data
    ###


#Load outDomainData
###Test
docs_head, docs_tail, data = load_GeneralDomain("data/opendomain_finetune_noword_10000/")
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

def in_Domain_Task_Data_binary(data_dir_indomain, tokenizer, max_seq_length):
    ###Open
    with open(data_dir_indomain+"train.json") as file:
        data = json.load(file)

    ###Preprocess
    num_label_list = list()
    label_sentence_dict = dict()
    for line in data:
        #line["sentence"]
        #line["aspect"]
        #line["sentiment"]
        num_label_list.append(line["aspect"])
        try:
            label_sentence_dict[line["aspect"]].append([line["sentence"]])
        except:
            label_sentence_dict[line["aspect"]] = [line["sentence"]]

    num_label = sorted(list(set(num_label_list)))
    label_map = {label : i for i , label in enumerate(num_label)}

    ###Create data: 1 choosed data along with the rest of 7 class data
    all_cur_tensors = list()
    for line in data:
        #line["sentence"]
        #line["aspect"]
        #line["sentiment"]
        sentence = line["sentence"]
        label = line["aspect"]
        sentence_out = [(random.choice(label_sentence_dict[label_out])[0], label_out) for label_out in num_label if label_out!=label]
        all_sentence = [(sentence, label)] + sentence_out #1st sentence is choosed

        all_input_ids = list()
        all_input_mask = list()
        all_segment_ids = list()
        all_lm_labels_ids = list()
        all_is_next = list()
        all_tail_idxs = list()
        all_sentence_labels = list()
        for id, sentence_label in enumerate(all_sentence):
            #tokens_a = tokenizer.tokenize(sentence_label[0])
            tokens_a = tokenizer.tokenize(sentence_label[0])
            '''
            if "</s>" in tokens_a:
                print("Have more than 1 </s>")
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
            all_sentence_labels.append(torch.tensor(label_map[sentence_label[1]]))

        cur_tensors = (torch.stack(all_input_ids),
                       torch.stack(all_input_ids_org),
                       torch.stack(all_input_mask),
                       torch.stack(all_segment_ids),
                       torch.stack(all_lm_labels_ids),
                       torch.stack(all_is_next),
                       torch.stack(all_tail_idxs),
                       torch.stack(all_sentence_labels))

        all_cur_tensors.append(cur_tensors)

    return all_cur_tensors


def load_outdomain(data_dir_outdomain, tokenizer, max_seq_length):
    ###Open
    doc_line = list()
    with open(data_dir_outdomain+"train.txt") as file:
        #data = json.load(file)
        for i,line in enumerate(file):
            doc_line.append(line)

    ###Preprocess

    cur_tensors_list=list()
    for i, line in enumerate(doc_line):

        tokens_a = tokenizer.tokenize(line)

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
                       torch.tensor(0),
                       torch.tensor(0))

        cur_tensors_list.append(cur_tensors)

    return cur_tensors_list


def in_Domain_Task_Data_binary(data_dir_indomain, tokenizer, max_seq_length):
    ###Open
    with open(data_dir_indomain+"train.json") as file:
        data = json.load(file)

    ###Preprocess
    num_label_list = list()
    label_sentence_dict = dict()
    for line in data:
        #line["sentence"]
        #line["aspect"]
        #line["sentiment"]
        num_label_list.append(line["aspect"])
        try:
            label_sentence_dict[line["aspect"]].append([line["sentence"]])
        except:
            label_sentence_dict[line["aspect"]] = [line["sentence"]]

    num_label = sorted(list(set(num_label_list)))
    label_map = {label : i for i , label in enumerate(num_label)}

    ###Create data: 1 choosed data along with the rest of 7 class data
    all_cur_tensors = list()
    for line in data:
        #line["sentence"]
        #line["aspect"]
        #line["sentiment"]
        sentence = line["sentence"]
        label = line["aspect"]
        sentence_out = [(random.choice(label_sentence_dict[label_out])[0], label_out) for label_out in num_label if label_out!=label]
        all_sentence = [(sentence, label)] + sentence_out #1st sentence is choosed

        all_input_ids = list()
        all_input_mask = list()
        all_segment_ids = list()
        all_lm_labels_ids = list()
        all_is_next = list()
        all_tail_idxs = list()
        all_sentence_labels = list()
        for id, sentence_label in enumerate(all_sentence):
            #tokens_a = tokenizer.tokenize(sentence_label[0])
            tokens_a = tokenizer.tokenize(sentence_label[0])
            '''
            if "</s>" in tokens_a:
                print("Have more than 1 </s>")
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
            all_sentence_labels.append(torch.tensor(label_map[sentence_label[1]]))

        cur_tensors = (torch.stack(all_input_ids),
                       torch.stack(all_input_ids_org),
                       torch.stack(all_input_mask),
                       torch.stack(all_segment_ids),
                       torch.stack(all_lm_labels_ids),
                       torch.stack(all_is_next),
                       torch.stack(all_tail_idxs),
                       torch.stack(all_sentence_labels))

        all_cur_tensors.append(cur_tensors)

    return all_cur_tensors



def AugmentationData_Domain(top_k, tokenizer, max_seq_length):
    #top_k_shape = top_k.indices.shape
    #ids = top_k.indices.reshape(top_k_shape[0]*top_k_shape[1]).tolist()
    top_k_shape = top_k["indices"].shape
    ids = top_k["indices"].reshape(top_k_shape[0]*top_k_shape[1]).tolist()

    all_input_ids = list()
    all_input_ids_org = list()
    all_input_mask = list()
    all_segment_ids = list()
    all_lm_labels_ids = list()
    all_is_next = list()
    all_tail_idxs = list()

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


    cur_tensors = (torch.stack(all_input_ids),
                   torch.stack(all_input_ids_org),
                   torch.stack(all_input_mask),
                   torch.stack(all_segment_ids),
                   torch.stack(all_lm_labels_ids),
                   torch.stack(all_is_next),
                   torch.stack(all_tail_idxs))

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


def AugmentationData_Task_pos_and_neg(top_k=None, tokenizer=None, max_seq_length=None, add_org=None, in_task_rep=None):

    input_ids_, input_ids_org_, input_mask_, segment_ids_, lm_label_ids_, is_next_, tail_idxs_, sentence_label_, sentiment_label_ = add_org


    all_sentence_binary_label = list()
    all_in_task_rep_comb = list()
    for id_1, num in enumerate(sentence_label_):
        sentence_label_int = (sentence_label_==num).to(torch.long)
        in_task_rep_append = in_task_rep[id_1].unsqueeze(0).expand(in_task_rep.shape[0],-1)
        in_task_rep_comb = torch.cat((in_task_rep_append,in_task_rep),-1)
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
                        '''
                        if line.strip() == "":
                            self.num_docs += 1
                        else:
                        '''
                        if line.strip() == "":
                            continue

                        self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() != "":
                        self.num_docs += 1

            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        #print(self.corpus_lines)
        #print(self.num_docs)
        #return self.corpus_lines - self.num_docs - 1
        return self.corpus_lines - self.num_docs

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
        tokens_a = self.tokenizer.tokenize(t1)
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
                       torch.tensor(0),
                       torch.tensor(cur_features.tail_idxs),
                       torch.tensor(0),
                       torch.tensor(0))
        '''
        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_ids_org),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.is_next),
                       torch.tensor(cur_features.tail_idxs))
        '''

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
    tail_idxs = len(input_ids)+1

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


    assert len(input_ids) == max_seq_length
    assert len(input_ids_org) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

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
                        default=None,
                        type=int,
                        required=True,
                        help="Choose Task")
    parser.add_argument("--K",
                        default=None,
                        type=int,
                        required=True,
                        help="K size")
    ####

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

    os.makedirs(args.output_dir, exist_ok=True)

    #tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model, do_lower_case=args.do_lower_case)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model)




    # Prepare model
    model = RobertaForMaskedLMDomainTask.from_pretrained(args.pretrain_model, output_hidden_states=True, return_dict=True, num_labels=args.num_labels_task)
    model.to(device)



    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)


    ###Generate Docs###
    docs_head = list()
    docs_tail = list()
    if True:
        print("Loading Train Dataset", args.data_dir_outdomain)
        train_dataset = Dataset_noNext(args.data_dir_outdomain, tokenizer, seq_len=args.max_seq_length, corpus_lines=None, on_memory=args.on_memory)
        #train_dataset = load_outdomain(args.data_dir_outdomain, tokenizer, args.max_seq_length)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    #logger.info("  Num steps = %d", num_train_optimization_steps)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)


    model.eval()
    #result = 0
    counter_domain = 0
    counter_task = 0
    id_doc = dict()
    all_doc_list = list()
    rest_head = list()
    rest_tail = list()
    rest_input = list()
    for step, batch_ in enumerate(tqdm(train_dataloader, desc="Iteration")):

        ###Normal mode
        batch_ = tuple(t.to(device) for t in batch_)
        input_ids_, input_ids_org_, input_mask_, segment_ids_, lm_label_ids_, is_next_, tail_idxs_, sentence_label_, sentiment_label_ = batch_

        '''
        print(input_ids_org_.shape)
        print(tokenizer.decode(input_ids_org_[0]).replace("<pad>",""))
        print(tokenizer.decode(input_ids_org_[0]).replace("<pad>","").replace("<s>","").replace("</s>",""))
        exit()
        '''

        #Generate query representation
        with torch.no_grad():
            in_domain_rep, in_task_rep = model(input_ids_org=input_ids_org_, tail_idxs=tail_idxs_, attention_mask=input_mask_, func="in_domain_task_rep")

            in_domain_rep = in_domain_rep.to("cpu")
            in_task_rep = in_domain_rep.to("cpu")


            for idx, ten in enumerate(in_domain_rep):
                docs_tail.append(ten)
                id_doc[counter_domain] = tokenizer.decode(input_ids_org_[idx]).replace("<pad>","").replace("<s>","").replace("</s>","")
                counter_domain+=1

            for ten in in_task_rep:
                docs_head.append(ten)
                counter_task+=1

            if counter_domain!=counter_task:
                print("Error")
                exit()

            #####
            #####
            '''
            try:
                docs_tail.append(in_domain_rep)
                docs_head.append(in_task_rep)
                all_doc_list.append(input_ids_org_)
            except:
                rest_tail = in_domain_rep
                rest_head = in_task_rep
                rest_input = input_ids_org_

            counter_domain += int(in_domain_rep.shape[0])
            counter_task += int(in_task_rep.shape[0])
            '''


    docs_tail = torch.stack(docs_tail).unsqueeze(1)
    docs_head = torch.stack(docs_head).unsqueeze(1)
    ###
    '''
    docs_tail_ = torch.stack(docs_tail[:-1])
    docs_tail = concate docs_tail[-1]
    docs_head_ = torch.stack(docs_head[:-1])
    docs_head =
    '''
    ###


    docs_tail = docs_tail.to("cpu")
    docs_head = docs_head.to("cpu")



    ###Retrive: Caculate || p_n and bottom_n###
    total_score = torch.zeros([args.train_batch_size,docs_head.shape[0]]).to("cpu")
    if True:
        print("Loading Train Dataset", args.data_dir_indomain)
        train_dataset = load_outdomain(args.data_dir_indomain, tokenizer, args.max_seq_length)
        #all_type_sentence, train_dataset = in_Domain_Task_Data_mutiple(args.data_dir_indomain, tokenizer, args.max_seq_length)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    #logger.info("  Num steps = %d", num_train_optimization_steps)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)


    model.eval()
    if args.K > docs_head.shape[0]:
        k = docs_head.shape[0]
    else:
        k = int(args.K)/2

    #########
    #####load_classifier
    #########
    with torch.no_grad():
        '''
        task_binary_classifier_weight, task_binary_classifier_bias = model(func="return_task_binary_classifier")
        task_binary_classifier_weight = task_binary_classifier_weight[:int(task_binary_classifier_weight.shape[0]/n_gpu)][:]
        task_binary_classifier_bias = task_binary_classifier_bias[:int(task_binary_classifier_bias.shape[0]/n_gpu)][:]
        task_binary_classifier = return_Classifier(task_binary_classifier_weight, task_binary_classifier_bias, 768*2, 2)
        '''
        domain_binary_classifier_weight, domain_binary_classifier_bias = model(func="return_domain_binary_classifier")
        domain_binary_classifier_weight = domain_binary_classifier_weight[:int(domain_binary_classifier_weight.shape[0]/n_gpu)][:]
        domain_binary_classifier_bias = domain_binary_classifier_bias[:int(domain_binary_classifier_bias.shape[0]/n_gpu)][:]
        domain_binary_classifier = return_Classifier(domain_binary_classifier_weight, domain_binary_classifier_bias, 768, 2)


    for step, batch_ in enumerate(tqdm(train_dataloader, desc="Iteration")):

        ###Normal mode
        batch_ = tuple(t.to(device) for t in batch_)
        input_ids_, input_ids_org_, input_mask_, segment_ids_, lm_label_ids_, is_next_, tail_idxs_, sentence_label_, sentiment_label_ = batch_

        #Generate query representation
        in_domain_rep, in_task_rep = model(input_ids_org=input_ids_org_, tail_idxs=tail_idxs_, attention_mask=input_mask_, func="in_domain_task_rep")

        ##Load classifier weight

        # Search id from Docs and ranking via (Domain/Task)
        query_domain = in_domain_rep.float().to("cpu")
        query_domain = query_domain.unsqueeze(1)
        query_task = in_task_rep.float().to("cpu")
        query_task = query_task.unsqueeze(1)


        query_domain = query_domain.expand(-1, docs_tail.shape[0], -1)
        query_task = query_task.expand(-1, docs_head.shape[0], -1)

        #################
        #################

        LeakyReLU = torch.nn.LeakyReLU()
        domain_binary_logit = LeakyReLU(domain_binary_classifier(docs_tail))
        domain_binary_logit = domain_binary_logit[:,:,1] - domain_binary_logit[:,:,0]
        domain_binary_logit = domain_binary_logit.squeeze(1).unsqueeze(0).expand(sentiment_label_.shape[0], -1)
        '''
        task_binary_logit = LeakyReLU(task_binary_classifier(torch.cat([query_task, docs_head[:,0,:].unsqueeze(0).expand(sentiment_label_.shape[0], -1, -1)], dim=2)))
        task_binary_logit = task_binary_logit[:,:,1] - task_binary_logit[:,:,0]
        '''

        '''
        results = domain_binary_logit + task_binary_logit
        '''
        results = domain_binary_logit

        total_score += results


    ########
    ########
    #sum all batch tensor
    total_score = total_score.sum(dim=0)
    #print(total_score.shape)

    #Ranking
    bottom_k = torch.topk(total_score, k, dim=0, largest=False, sorted=False)
    bottom_k = {"values":bottom_k.values, "indices":bottom_k.indices}
    top_k = torch.topk(total_score, k, dim=0, largest=True, sorted=False)
    top_k = {"values":top_k.values, "indices":top_k.indices}

    #print(bottom_k["indices"].shape)
    #print(top_k["indices"].shape)

    choosed_docs = torch.cat([top_k["indices"],bottom_k["indices"]],0)
    #print(choosed_docs.shape)

    #select 1.tensor 2.text from foc_head,doc_tail
    all_data_dict = dict()
    head_hidd_list = list()
    tail_hidd_list = list()
    #print(len(id_doc))
    for id, index in enumerate(choosed_docs):
        all_data_dict[id] = {"sentence":id_doc[int(index)]}
        head_hidd_list.append(docs_head[index])
        tail_hidd_list.append(docs_tail[index])

    #exit()

    ############################################
    ############################################
    with open(args.output_dir+'.json', 'w') as outfile:
        json.dump(all_data_dict, outfile)

        head_hidd_tensor = torch.stack(head_hidd_list)
        tail_hidd_tensor = torch.stack(tail_hidd_list)


    torch.save(head_hidd_tensor, args.output_dir+'_head.pt')
    torch.save(tail_hidd_tensor, args.output_dir+'_tail.pt')







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
