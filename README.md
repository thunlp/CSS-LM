# CSS-LM
[CSS-LM](https://arxiv.org/pdf/2102.03752.pdf): Contrastive Semi-supervised Fine-tuning of Pre-trained Language Models

Overview
=============
![CSS-LM](https://github.com/thunlp/CSS-LM/blob/main/CSS-LM.jpg)



<figure>
<img src=https://github.com/thunlp/CokeBERT/blob/main/CSS-LM.jpg width="80%">
<figcaption>CSS-LM improves the fine-tuning phase of PLMs via contrastive semi-supervised learning. Specifically, given a specific task, we retrieve positive and negative instances from large-scale unlabeled corpora according to their domain-level and class-level semantic relatedness to the task. By performing contrastive semi-supervised learning on both the retrieved unlabeled and original labeled instances, CSS-LM can help PLMs capture crucial task-related semantic features and achieve better performance in low-resource scenarios.
</figcaption>
</figure>


Requirements and Prepare the data
=============
```
bash requirement.sh
```

Run CSS-LM
=============
By executing run1.sh, the code will automatically create folders for the corresponding datasets to save the checkpoints.
```
cd script
bash run1.sh
```
(You can refer to run1.sh for more details.)



Citation
=============
Please cite our paper if you use LM-BFF in your work:
```
@inproceedings{su2021csslm,
   title={CSS-LM: A Contrastive Framework for Semi-supervised Fine-tuning of Pre-trained Language Models},
   author={Su, Yusheng and Han, Xu and Lin, Yankai and Zhang, Zhengyan and Liu, Zhiyuan and Li, Peng and Zhou, Jie and  Sun, Maosong},
   booktitle={Association for WWW Workshop},
   year={2021}
}
```






