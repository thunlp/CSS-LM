gpu_0=4
gpu_1=5
gpu_2=6
gpu_3=7
gpu_4=4

#N_1=16
#N_1=8
#N_1=24
N_1=all
#N_1=30
N_2=24
N_3=32

#N_times_1=23
#N_times_2=23
#N_times_1=4
#N_times_2=4
#N_times_1=2
#N_times_2=2
N_times_1=5
N_times_2=5

batch_size=2

#comment eval_text --> when change to 2, there is a space error. files haven't fix in comment part.


#all --> use batch_size=8
#3

bash retrieve_semeval.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size



