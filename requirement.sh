pip install -r requirement.txt

git clone git@github.com:NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

#wget data
#https://cloud.tsinghua.edu.cn/d/f6a19d5373e74fbfb42d/

scp -r download/opendomain_finetune_noword_10000 data
scp -r download/openwebtext data
scp -r download/roberta-base-768 script
scp -r download/bert-base-768 script

