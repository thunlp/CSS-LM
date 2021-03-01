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

batch_size=8

#comment eval_text --> when change to 2, there is a space error. files havent fix in comment part.


#all --> use batch_size=8
#3

'''
#bash run_semeval_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
#bash run_semeval_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
bash run_semeval_sscl_dt.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size



#5
bash run_sst5_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
bash run_sst5_sscl_dt.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
bash run_sst5_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size



#######
#3
bash run_scicite_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
bash run_scicite_sscl_dt.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
bash run_scicite_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size

'''


#12
#bash run_chemprot_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
#bash run_chemprot_sscl_dt.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
bash run_chemprot_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size



'''
#6
bash run_aclintent_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
bash run_aclintent_sscl_dt.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
bash run_aclintent_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size




#7
bash run_sciie_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
bash run_sciie_sscl_dt.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
bash run_sciie_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size
'''




###############################
###############################

