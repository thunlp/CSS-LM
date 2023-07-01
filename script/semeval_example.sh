gpu_0=0
gpu_1=1
gpu_2=2
gpu_3=3
gpu_4=4
gpu_5=5
gpu_6=6
gpu_7=7

N_1=16
N_2=24
N_3=32

N_times_1=14
N_times_2=14

#All reduce to 16 retrieved instances
batch_size=4
max_length=100

#all --> use batch_size=8
#3
for i_th in {1..5};
do
    #RoBERTa-base Model
    bash run_semeval_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_semeval_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th

    #BERT-base Moodel
    bash run_bert_semeval_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_semeval_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th

done

