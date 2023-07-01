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

## Run the Experiments
Excute 'script/run1.sh'.
```bash
cd script
bash run1.sh
```

In `run1.sh`, we have two kinds of backbone models (`BERT` and `RoBERTa`). 
- run_{$DATASET}_finetune.sh: Few-shot Fine-tuning
- run_{$DATASET}_sscl_dt_k.sh: 
- run_{$DATASET}_st.sh:
- run_{$DATASET}_sscl.sh:

- run_bert_{$DATASET}_finetune.sh:
- run_bert_{$DATASET}_finetune.sh:
- run_bert_{$DATASET}_finetune.sh:
- run_bert_{$DATASET}_finetune.sh:

```bash
for i_th in {1..5};
do
    #RoBERTa-base Model
    bash run_semeval_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_semeval_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_semeval_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_semeval_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th

    #BERT-base Moodel
    bash run_bert_semeval_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_semeval_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_semeval_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_semeval_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done
```



## Run CSS-LM

By executing run1.sh, the code will automatically create folders for the corresponding datasets to save the checkpoints.
```
cd script
bash run1.sh
```
(You can refer to run1.sh for more details.)



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




