#install package
pip install -r requirement.txt

#install apex
git clone git@github.com:NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

#wget data:
#https://cloud.tsinghua.edu.cn/f/690e78d324ee44068857/?dl=1

wget https://cloud.tsinghua.edu.cn/f/690e78d324ee44068857/?dl=1

mv 'index.html?dl=1' download.zip
unzip download.zip download

scp -r download/opendomain_finetune_noword_10000 data
scp -r download/openwebtext data
scp -r download/roberta-base-768 script
scp -r download/bert-base-768 script

