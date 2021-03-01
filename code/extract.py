import os
import sys
import csv

dir_input = sys.argv[1]
dir_output = sys.argv[2]
N = sys.argv[3]
mode = sys.argv[4]
if type(N) == int:
    N = str(N)


fileDir = os.listdir(dir_input)
if str(mode) == "test":
    fileDir = [file for file in fileDir if str(mode)+"_paper_results_" in file]
elif str(mode) == "eval":
    fileDir = [file for file in fileDir if str(mode)+"_results_" in file]


#best_p = 0
#best_r = 0
best_f1 = 0
best_model = 0
best_acc = 0
best_loss = 0
for file in fileDir:
    if (str(mode) == "test") and ("_best" not in file):
        with open(dir_input+"/"+file) as f:
            for line in f:
                line = line.strip().split()
                if line[0]=='F1':
                    score = float(line[-1].replace("%",""))/100
                    if score > best_f1:
                        best_f1 = score
                        best_model = file
                else:
                    pass
    elif (str(mode) == "eval") and ("_best" not in file):
        with open(dir_input+"/"+file) as f:
            for line in f:
                line = line.strip().split()
                if line[0]=='eval_accuracy':
                    acc = float(line[-1])
                #if line[0]=='eval_loss':
                #    loss = float(line[-1])
                if acc > best_acc and "_test_paper_results_best.txt" not in file:
                    best_acc = acc
                    best_model = file

save_list = list()
with open(str(dir_input)+"/"+str(best_model)) as fin:
    for line in fin:
        save_list.append([line.strip()])

with open(str(dir_output)+"/"+str(N)+"_"+str(best_model), "w", newline='\n') as fout:
    writer = csv.writer(fout)
    writer.writerows(save_list)

