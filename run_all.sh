gpu_0=4
gpu_1=5
gpu_2=6
gpu_3=7
gpu_4=4
gpu_5=5
gpu_6=6
gpu_7=7

#N_1=16
N_1="all"
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
N_times_1=6
N_times_2=6
#N_times_1=14
#N_times_2=14

batch_size=4
#max_length=128
max_length=100

#comment eval_text --> when change to 2, there is a space error. files haven't fix in comment part.


#all --> use batch_size=8
#3
#for i_th in {1..5};
for i_th in "all";
do
    '''
    bash run_semeval_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_semeval_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_semeval_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_semeval_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    '''

    ######################
    ########BERT##########
    ######################
    #bash run_bert_semeval_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #bash run_bert_semeval_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th

    bash run_bert_semeval_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th

    #bash run_bert_semeval_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done





#5
#for i_th in {1..5};
for i_th in "all";
do
    '''
    bash run_sst5_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_sst5_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_sst5_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_sst5_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    '''
    ######################
    ########BERT##########
    ######################
    #bash run_bert_sst5_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_sst5_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_sst5_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_sst5_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done

#####
#####


#######
#3
#for i_th in {1..5};
for i_th in "all";
do
    '''
    bash run_scicite_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_scicite_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_scicite_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_scicite_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    '''
    ######################
    ########BERT##########
    ######################
    #bash run_bert_scicite_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    #bash run_bert_scicite_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_scicite_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_scicite_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done



#6
#for i_th in {1..5};
for i_th in "all";
do
    '''
    bash run_aclintent_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_aclintent_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_aclintent_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_aclintent_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    '''
    ######################
    ########BERT##########
    ######################
    bash run_bert_aclintent_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_aclintent_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_aclintent_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_aclintent_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done




#7
#for i_th in {1..5};
for i_th in "all";
do
    '''
    bash run_sciie_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_sciie_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_sciie_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_sciie_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    '''
    ######################
    ########BERT##########
    ######################
    bash run_bert_sciie_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_sciie_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_sciie_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_sciie_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done



#12
#for i_th in {1..5};
for i_th in "all";
do
    '''
    bash run_chemprot_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_chemprot_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_chemprot_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_chemprot_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    '''
    ######################
    ########BERT##########
    ######################
    bash run_bert_chemprot_finetune_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_chemprot_sscl_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_chemprot_sscl_dt_k_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
    bash run_bert_chemprot_st_all.sh $gpu_0 $gpu_1 $gpu_2 $gpu_3 $N_1 $N_2 $N_3 $N_times_1 $N_times_2 $batch_size $max_length $i_th
done



