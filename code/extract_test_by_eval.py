import os
import sys
import csv

dir_input = sys.argv[1]
dir_output = sys.argv[2]
N = sys.argv[3]
#mode = sys.argv[4]
if type(N) == int:
    N = str(N)


fileDir = os.listdir(dir_input)
#if str(mode) == "test":
#    fileDir = [file for file in fileDir if "test_paper_results_" in file]
#elif str(mode) == "eval":
#    fileDir = [file for file in fileDir if "eval_results_" in file]
fileDir = [file for file in fileDir]


#best_p = 0
#best_r = 0
best_f1 = 0
best_model = 0
best_model_eval = 0
best_acc = 0
best_loss = 0

#Find best eval
for file in fileDir:
    if "eval" in file:
        with open(dir_input+"/"+file) as f:
            for line in f:
                line = line.strip().split()
                #if line[0]=='eval_accuracy':
                if line[0]=='eval_accuracy':
                    acc = float(line[-1])
                    #loss = float(line[-1])
                    if acc >= best_acc and "_test_paper_results_best.txt" not in file:
                        best_acc = acc
                        best_model_eval = file


save_list = list()
with open(str(dir_input)+"/"+str(best_model_eval)) as fin:
    for line in fin:
        save_list.append([line.strip()])

with open(str(dir_output)+"/"+str(N)+"_"+str(best_model_eval), "w", newline='\n') as fout:
    writer = csv.writer(fout)
    writer.writerows(save_list)



#Test in the best eval
for file in fileDir:
    if "test_paper_results_" in file:
        if file.split("_")[-1] == best_model_eval.split("_")[-1]:
            with open(dir_input+"/"+file) as f:
                #print(file.split("_")[-1])
                #print(best_model_eval.split("_")[-1])
                #if file.split("_")[-1] == best_model_eval.split("_")[-1]:
                #best_f1 = score
                #best_model = file
                for line in f:
                    line = line.strip().split()
                    if line[0]=='F1':
                        score = float(line[-1].replace("%",""))/100
                        if score > best_f1:
                            best_f1 = score
                            best_model = file
                    else:
                        pass


save_list = list()
with open(str(dir_input)+"/"+str(best_model)) as fin:
    for line in fin:
        save_list.append([line.strip()])

with open(str(dir_output)+"/"+str(N)+"_"+str(best_model), "w", newline='\n') as fout:
    writer = csv.writer(fout)
    writer.writerows(save_list)

###############################
###############################
###############################

best_model_test = 0
score=0
best_f1=0

#find the best test
for file in fileDir:
    if "test_paper_results_" in file:
        with open(dir_input+"/"+file) as f:
            for line in f:
                line = line.strip().split()
                if line[0]=='F1':
                    score = float(line[-1].replace("%",""))/100
                    if score > best_f1:
                        best_f1 = score
                        best_model_test = file
                else:
                    pass

with open(str(dir_input)+"/"+str(best_model_test)) as fin:
    for line in fin:
        save_list.append([line.strip()])

with open(str(dir_output)+"/"+str(N)+"_"+str(best_model_test)+"_no_eval", "w", newline='\n') as fout:
    writer = csv.writer(fout)
    writer.writerows(save_list)
