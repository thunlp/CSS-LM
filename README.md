# CSS-LM
[CSS-LM](https://arxiv.org/abs/2102.03752): A Contrastive Framework for Semi-supervised Fine-tuning of Pre-trained Language Models

## Overview

![CSS-LM](https://github.com/thunlp/CSS-LM/blob/main/CSS-LM.jpg)
CSS-LM improves the fine-tuning phase of PLMs via contrastive semi-supervised learning. Specifically, given a specific task, we retrieve positive and negative instances from large-scale unlabeled corpora according to their domain-level and class-level semantic relatedness to the task. By performing contrastive semi-supervised learning on both the retrieved unlabeled and original labeled instances, CSS-LM can help PLMs capture crucial task-related semantic features and achieve better performance in low-resource scenarios.



## Requirements and Prepare the data

```
bash requirement.sh
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
@article{2021,
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


@inproceedings{su2021csslm,
   title={CSS-LM: A Contrastive Framework for Semi-supervised Fine-tuning of Pre-trained Language Models},
   author={Su, Yusheng and Han, Xu and Lin, Yankai and Zhang, Zhengyan and Liu, Zhiyuan and Li, Peng and Zhou, Jie and  Sun, Maosong},
   booktitle={Proceedings of WWW Workshop},
   year={2021}
}
```


## Contact
[Yusheng Su](https://yushengsu-thu.github.io/)

Mail: yushengsu.thu@gmail.com; suys19@mauls.tsinghua.edu.cn




