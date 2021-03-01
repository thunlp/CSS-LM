gpu_0=0
gpu_1=1
gpu_2=2
gpu_3=3
gpu_4=4
gpu_5=5
gpu_6=6
gpu_7=7

N_1=16
#N_1=8
#N_1=24
#N_1=all
#N_1=30
N_2=24
N_3=32

#N_times_1=23
#N_times_2=23
#N_times_1=1
#N_times_2=1
N_times_1=8
N_times_2=8
#N_times_1=14
#N_times_2=14

#batch_size=4
batch_size=16
#max_length=128
max_length=100

#comment eval_text --> when change to 2, there is a space error. files haven't fix in comment part.






#5
bash run_sst5_finetune_roberta_bert.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length
