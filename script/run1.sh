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
#N_1=32
#N_1=all
#N_1=30
N_2=24
N_3=32

#N_times_1=23
#N_times_2=23
#N_times_1=2
#N_times_2=2
#N_times_1=8
#N_times_2=8
#N_times_1=12
#N_times_2=12
N_times_1=14
N_times_2=14

#All reduce to 16 retrieved instances

batch_size=4
max_length=100


#all --> use batch_size=8
#3
for i_th in {1..5};
do
    bash run_semeval_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    exit
    #add back 32,48
    #bash run_semeval_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #add back 32,48
    #bash run_semeval_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_semeval_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th

    ######################
    ########BERT##########
    ######################
    bash run_bert_semeval_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #add back 32,48
    bash run_bert_semeval_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #add back 32,48
    bash run_bert_semeval_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_semeval_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done





#5
for i_th in {1..5};
do
    bash run_sst5_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_sst5_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_sst5_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_sst5_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    ######################
    ########BERT##########
    ######################
    bash run_bert_sst5_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_sst5_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_sst5_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_sst5_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done

#####
#####


#######
#3
for i_th in {1..5};
do
    bash run_scicite_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #add back 32,48
    bash run_scicite_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #add back 32,48
    bash run_scicite_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_scicite_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    ######################
    ########BERT##########
    ######################
    bash run_bert_scicite_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #add back 32,48
    bash run_bert_scicite_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #add back 32,48
    bash run_bert_scicite_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_scicite_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done



#6
for i_th in {1..5};
do
    bash run_aclintent_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_aclintent_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_aclintent_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_aclintent_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    ######################
    ########BERT##########
    ######################
    bash run_bert_aclintent_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_aclintent_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_aclintent_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_aclintent_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done




#7
for i_th in {1..5};
do
    bash run_sciie_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #add back 32,48
    bash run_sciie_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #add back 32,48
    bash run_sciie_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_sciie_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    ######################
    ########BERT##########
    ######################
    bash run_bert_sciie_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #add back 32,48
    bash run_bert_sciie_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #add back 32,48
    bash run_bert_sciie_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_sciie_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done



#12
for i_th in {1..5};
do
    bash run_chemprot_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_chemprot_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_chemprot_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_chemprot_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    ######################
    ########BERT##########
    ######################
    bash run_bert_chemprot_finetune.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_chemprot_sscl_dt_k.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_chemprot_st.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_chemprot_sscl.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done



