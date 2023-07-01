# CSS-LM
[CSS-LM](https://arxiv.org/abs/2102.03752): A Contrastive Framework for Semi-supervised Fine-tuning of Pre-trained Language Models

- WWW-Workshop 2021 Accepted.

- IEEE/TASLP 2021 Accepted.

## Overview

![CSS-LM](https://github.com/thunlp/CSS-LM/blob/main/CSS-LM.jpg)
CSS-LM improves the fine-tuning phase of PLMs via contrastive semi-supervised learning. Specifically, given a specific task, we retrieve positive and negative instances from large-scale unlabeled corpora according to their domain-level and class-level semantic relatedness to the task. By performing contrastive semi-supervised learning on both the retrieved unlabeled and original labeled instances, CSS-LM can help PLMs capture crucial task-related semantic features and achieve better performance in low-resource scenarios.

## Setups
- python>=3.6
- torch>=2.0.0+cu118


## Requirements 
```
pip install -r requirement.sh
```

<!--
```
git clone git@github.com:NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```
-->






## Prepare the data
Download the open domain corpus (`openwebtext`) and backbone models (`roberta-base`, `bert-base-uncased`) and move them to the corresponding directories.
```bash
wget https://cloud.tsinghua.edu.cn/f/690e78d324ee44068857/?dl=1
mv 'index.html?dl=1' download.zip
unzip download.zip

rm -rf __MACOSX
scp -r download/openwebtext data
scp -r download/roberta-base script/roberta-base-768
scp -r download/bert-base-uncased script/bert-base-768
```
<!-- scp -r download/opendomain_finetune_noword_10000 data-->

## Run the All Experiments
Excute 'script/run1.sh'.
```bash
cd script
bash run1.sh
```

In `run1.sh`, we have two kinds of backbone models (`BERT` and `RoBERTa`). 
### RoBERTa-based 
- run_${DATASET}_finetune.sh: Few-shot Fine-tuning (Called <b>Standard</b> in the paper.)
- run_${DATASET}_sscl_dt_k.sh: Semi-supervised Contrastive Fine-tuning (Called <b>CSS-LM</b> in the paper.)
- run_${DATASET}_st.sh: Supervised Contrastive Fine-tuning (Called <b>SCF</b> in the paper.) 
- run_${DATASET}_sscl.sh: Semi-supervised Contrastive Pseudo Labeling Fine-tuning (Called <b>CSS-LM-ST</b> in the paper.)

### BERT-based 
- run_bert_${DATASET}_finetune.sh: Few-shot Fine-tuning (Called <b>Standard</b> in the paper.)
- run_bert_${DATASET}_finetune.sh: Semi-supervised Contrastive Fine-tuning (Called <b>CSS-LM</b> in the paper.)
- run_bert_${DATASET}_finetune.sh: Supervised Contrastive Fine-tuning (Called <b>SCF</b> in the paper.)
- run_bert_${DATASET}_finetune.sh: Semi-supervised Contrastive Pseudo Labeling Fine-tuning (Called <b>CSS-LM-ST</b> in the paper.)

`${DATASET}`: Can be semeval, sst5, scicite, aclintent, sciie, chemprot, and chemprot.
`$gpu_0 $gpu_1 $gpu_2 $gpu_3`: You could assign the numbers of GPUs and gpu_ids that you need.
`$N_1 $N_2 $N_3`: The number of annotated instances.
`$N_times_1 $N_times_2`: The number of training epoches.
`$batch_size`: Training batch size.
`$max_length`: The max length of the input sentence.
`$i_th`: Given 5 random seeds to train the models. Each `$i_th` indicates the different random seed.

```bash
for i_th in {1..5};
do
    #RoBERTa-based Model
    bash run_${DATASET}_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_${DATASET}_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_${DATASET}_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_${DATASET}_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th

    #BERT-based Moodel
    bash run_bert_${DATASET}_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_${DATASET}_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_${DATASET}_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_${DATASET}_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done
```


<!--
## Run CSS-LM

By executing run1.sh, the code will automatically create folders for the corresponding datasets to save the checkpoints.
```
cd script
bash run1.sh
```
(You can refer to run1.sh for more details.)
-->



## Citation

Please cite our paper if you use CSS-LM in your work:
```
@article{su2021csslm,
   title={CSS-LM: A Contrastive Framework for Semi-Supervised Fine-Tuning of Pre-Trained Language Models},
   volume={29},
   ISSN={2329-9304},
   url={http://dx.doi.org/10.1109/TASLP.2021.3105013},
   DOI={10.1109/taslp.2021.3105013},
   journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Su, Yusheng and Han, Xu and Lin, Yankai and Zhang, Zhengyan and Liu, Zhiyuan and Li, Peng and Zhou, Jie and Sun, Maosong},
   year={2021},
   pages={2930–2941}
}
```


## Contact
[Yusheng Su](https://yushengsu-thu.github.io/)

Mail: yushengsu.thu@gmail.com; suys19@mauls.tsinghua.edu.cn




