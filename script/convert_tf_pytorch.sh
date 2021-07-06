
DIR_IN=/data5/private/suyusheng/task_selecte/uncased_L-12_H-128_A-2
DIR_OUT=/data5/private/suyusheng/task_selecte/bert-base-uncased-128

cp $DIR_IN/bert_config.json $DIR_OUT/config.json
cp $DIR_IN/vocab.txt $DIR_OUT/vocab.txt

python3 /data5/private/suyusheng/task_selecte/code/convert_bert_original_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path  $DIR_IN/bert_model.ckpt \
  --bert_config_file  $DIR_OUT/config.json  \
  --pytorch_dump_path  $DIR_OUT/pytorch_model.bin \
