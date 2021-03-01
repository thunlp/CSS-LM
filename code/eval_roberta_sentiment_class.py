import argparse
import logging
import random
import numpy as np
import os
import json

import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification
from transformers.modeling_roberta import RobertaForMaskedLMDomainTask
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.optimization import AdamW, get_linear_schedule_with_warmup



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids=None, attention_mask=None, segment_ids=None, label_id=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, sentence, aspect, sentiment=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.sentence = sentence
        self.aspect = aspect
        self.sentiment = sentiment


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            return json.loads(f.read())


class Processor_1(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        aspect = set([x.aspect for x in examples])
        sentiment = set([x.sentiment for x in examples])
        return examples, list(aspect), list(sentiment)

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")
        aspect = set([x.aspect for x in examples])
        sentiment = set([x.sentiment for x in examples])
        return examples, list(aspect), list(sentiment)

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")
        aspect = set([x.aspect for x in examples])
        sentiment = set([x.sentiment for x in examples])
        return examples, list(aspect), list(sentiment)

    def get_labels(self):
        """Useless"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)

            sentence = line["sentence"]
            aspect = line["aspect"]
            sentiment = line["sentiment"]

            examples.append(
                InputExample(guid=guid, sentence=sentence, aspect=aspect, sentiment=sentiment))
        return examples

def convert_examples_to_features(examples, aspect_list, sentiment_list, max_seq_length, tokenizer, task_n):

    """Loads a data file into a list of `InputBatch`s."""

    #Task_1: sentence --> aspect
    #Task_2: aspect+sentence --> sentiment
    if task_n == 1:
        label_list = sorted(aspect_list)
    elif task_n == 2:
        label_list = sorted(sentiment_list)
    else:
        print("Wrong task")
    '''
    for w in label_list:
        print(w,tokenizer.encode(w))
    exit()
    '''
    label_map = {label : i for i, label in enumerate(label_list)}


    features = []
    for (ex_index, example) in enumerate(examples):

        #Add new special tokens
        '''
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
        special_tokens_dict = {'cls_token': '<CLS>'}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print('We have added', num_added_toks, 'tokens')
        model.resize_token_embeddings(len(tokenizer))
        '''

        '''
        print(tokenizer.all_special_tokens)
        print(tokenizer.encode(tokenizer.all_special_tokens))
        #['[PAD]', '[SEP]', '[CLS]', '[MASK]', '[UNK]']
        #[ 0, 102, 101, 103, 100]
        '''


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

        ###
        #Already add [CLS] and [SEP]
        #101, 102
        input_ids = tokenizer.encode(example.sentence,add_special_tokens=True)
        ###
        next_input = tokenizer.encode(example.aspect,add_special_tokens=False)
        #next_input = tokenizer.encode("restaurant",add_special_tokens=False)
        next_input = [3] + next_input + [2]
        input_ids+=next_input
        ###
        segment_ids = [0] * len(input_ids)

        '''
        if task_n==2:
            #"[SEP]"
            input_ids += input_ids + [102]
            #sentiment: word (Next sentence)
            #segment_ids += [1] * (len(tokens_b) + 1)
        '''

        # The “Attention Mask” is simply an array of 1s and 0s indicating which tokens are padding and which aren’t (including special tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding
        segment_ids += padding

        if len(input_ids) != max_seq_length:
            print("input_ids more than max_seq_length")
            continue
        assert len(input_ids) == max_seq_length
        if len(attention_mask) != max_seq_length:
            print("attention_mask more than max_seq_length")
            continue
        assert len(attention_mask) == max_seq_length
        if len(segment_ids) != max_seq_length:
            print("segment_ids more than max_seq_length")
            continue
        assert len(segment_ids) == max_seq_length


        if task_n == 1:
            label_id = label_map[example.aspect]
        elif task_n == 2:
            label_id = label_map[example.sentiment]
        else:
            print("Wrong task")


        if task_n == 1:
            features.append(
                    InputFeatures(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  segment_ids=None,
                                  label_id=label_id))
        elif task_n == 2:
            features.append(
                    InputFeatures(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id))
        else:
            print("Wrong in convert_examples")


    return features


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    ###############
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--pretrain_model",
                        default='bert-case-uncased',
                        type=str,
                        required=True,
                        help="Pre-trained model")
    parser.add_argument("--num_labels_task",
                        default=None, type=int,
                        required=True,
                        help="num_labels_task")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
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
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
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
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
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
    ###############

    args = parser.parse_args()
    #print(args.do_train, args.do_eval)
    #exit()

    processors = Processor_1

    num_labels = args.num_labels_task

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))



    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    #args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    '''
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    '''
    os.makedirs(args.output_dir, exist_ok=True)



    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model)


    train_examples = None
    num_train_steps = None
    aspect_list = None
    sentiment_list = None
    processor = processors()
    num_labels = num_labels
    #train_examples, aspect_list, sentiment_list = processor.get_train_examples(args.data_dir)

    filenames = os.listdir(args.output_dir)
    filenames = [x for x in filenames if "pytorch_model.bin_" in x]

    file_mark = []
    for x in filenames:
        file_mark.append([x, True])
        file_mark.append([x, False])

    ####
    ####
    test_examples, aspect_list, sentiment_list = processor.get_test_examples(args.data_dir)
    if args.task == 1:
        num_labels = len(aspect_list)
    elif args.task == 2:
        num_labels = len(sentiment_list)
    else:
        print("What's task?")
        exit()
    test = convert_examples_to_features(
        test_examples, aspect_list, sentiment_list, args.max_seq_length, tokenizer, args.task)
    eval_examples = test_examples
    ###


    for x, mark in file_mark:
        print(x, mark)
        output_model_file = os.path.join(args.output_dir, x)
        print(output_model_file)

        model = RobertaForSequenceClassification.from_pretrained(args.pretrain_model, num_labels=num_labels, output_hidden_states=False, output_attentions=False, return_dict=True)
        model.load_state_dict(torch.load(output_model_file), strict=False)
        #strict False: ignore non-matching keys


        #param_optimizer = [para[0] for para in model.named_parameters()]
        #param_optimizer = [para for para in model.named_parameters()][-2]
        #print(param_optimizer)

        model.to(device)
        if mark:
            eval_features = dev
        else:
            eval_features = test

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)


        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
        if args.task == 1:
            print("Excuting the task 1")
        elif args.task == 2:
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        else:
            print("Wrong here2")

        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        if args.task == 1:
            eval_data = TensorDataset(all_input_ids, all_attention_mask, all_label_ids)
        elif args.task == 2:
            eval_data = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
        else:
            print("Wrong here1")

        if args.local_rank == -1:
            eval_sampler = RandomSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if mark:
            output_eval_file = os.path.join(args.output_dir, "eval_results_{}.txt".format(x.split("_")[-1]))
            output_file_pred = os.path.join(args.output_dir, "eval_pred_{}.txt".format(x.split("_")[-1]))
            output_file_glod = os.path.join(args.output_dir, "eval_gold_{}.txt".format(x.split("_")[-1]))
        else:
            output_eval_file = os.path.join(args.output_dir, "test_results_{}.txt".format(x.split("_")[-1]))
            output_file_pred = os.path.join(args.output_dir, "test_pred_{}.txt".format(x.split("_")[-1]))
            output_file_glod = os.path.join(args.output_dir, "test_gold_{}.txt".format(x.split("_")[-1]))

        fpred = open(output_file_pred, "w")
        fgold = open(output_file_glod, "w")

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0


        for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
            #batch = tuple(t.to(device) if i != 3 else t for i, t in enumerate(batch))
            batch = tuple(t.to(device) for i, t in enumerate(batch))

            if args.task == 1:
                input_ids, attention_mask, label_ids = batch
            elif args.task == 2:
                input_ids, attention_mask, segment_ids, label_ids = batch
            else:
                print("Wrong here3")


            if args.task == 1:
                #loss, logits, hidden_states, attentions
                output = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=label_ids)
                logits = output.logits
                tmp_eval_loss = output.loss
            elif args.task == 2:
                #loss, logits, hidden_states, attentions
                output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask, labels=label_ids)
                logits = output.logits
                tmp_eval_loss = output.loss
            else:
                print("Wrong!!")


            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy, pred = accuracy(logits, label_ids)
            for a, b in zip(pred, label_ids):
                fgold.write("{}\n".format(b))
                fpred.write("{}\n".format(a))

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy
                  }

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))



if __name__ == "__main__":
    main()


