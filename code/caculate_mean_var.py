import os
import sys
import csv
from statistics import mean, pstdev

#dir_input = sys.argv[1]
#dir_output = sys.argv[2]
#N = sys.argv[3]
#mode = sys.argv[4]
#if type(N) == int:
#    N = str(N)

#dataset
#datasets = {"semeval":"","sst5":"_sst5","scicite":"_scicite","aclintent":"_aclintent","sciie":"_sciie","chemprot":"_chemprot"}
datasets = {"semeval":"","scicite":"_scicite","sciie":"_sciie"}

#datasets = {"semeval":""}
#datasets = {"sst5":"_sst5"}
#datasets = {"scicite":"_scicite"}
#datasets = {"aclintent":"_aclintent"}
#datasets = {"sciie":"_sciie"}
#datasets = {"chemprot":"_chemprot"}

#mode
modes = ["roberta","bert"]
#modes = ["roberta"]
#modes = ["bert"]

#times
times_list=list(range(1,6)) #1~5
#times_list=list(range(1,3))

#supervised_data
#supervised_data=["16","24","32","48"]
#supervised_data=["16"]
#supervised_data=["24"]
supervised_data=["32"]


for dataset in datasets:

    #finetune
    ###################################
    ###################################
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!! Dataset:",dataset,"!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("#################")
    print("Model:","Finetune")
    print("#################")
    print("-----------------")

    for mode in modes:
        test_i=list()
        eval_i=list()

        if mode == "roberta":
            dir_input = "baseline_n_result_open_domain"+str(datasets[dataset])+"_entropy_finetune"
            #print(dir_input)
            #print("--")
        elif mode == "bert":
            dir_input = "baseline_bert_n_result_open_domain"+str(datasets[dataset])+"_entropy_finetune"
            #print(dir_input)
            #print("==")

        #1~5
        for i in times_list: #1~5
            file_dir = dir_input+"_"+str(i)
            try:
                fileDir = os.listdir(file_dir)
                #find file
                test_file = [f for f in fileDir if "test_paper_results" in f and "txt_no_eval" not in f and supervised_data[0] in f][0]
                eval_file = [f for f in fileDir if "eval_results" in f and supervised_data[0] in f][0]
                #test
                with open(file_dir+"/"+test_file) as f:
                    for line in f:
                        line = line.strip().split()
                        if line[0] == "F1":
                            test_i.append(float(line[-1].replace("%","")))
                #eval
                with open(file_dir+"/"+eval_file) as f:
                    for line in f:
                        line = line.strip().split()
                        if line[0] == "eval_accuracy":
                            eval_i.append(float(line[-1])*100)
                print("File:", file_dir, ": DONE")
            except:
                print("File:", file_dir, ": PENDING...")
                #print("No file")
                #print("Less than 5 times")
                pass

        ####Caculate
        print("Mode:",mode)
        try:
            mean_test_i = mean(test_i)
            mean_eval_i = mean(eval_i)
            std_test_i = pstdev(test_i)
            std_eval_i = pstdev(eval_i)
            print("test:","{:.2f}%".format(mean_test_i)," ; ","std:","{:.2f}".format(std_test_i))
            print("eval:","{:.2f}%".format(mean_eval_i)," ; ","std:","{:.2f}".format(std_eval_i))
            print("test:",test_i)
            print("eval:",eval_i)
            print("------")
        except:
            print("test:","Pending...")
            print("eval:","Pending...")
            print("------")
    print("-------------------------------------------")
    ###################################
    ###################################



    #sscl
    ###################################
    ###################################
    #print("Dataset:",dataset)
    #print("!!!!!!!!!!!!!!!!!")
    print("#################")
    print("Model:","sscl")
    print("#################")
    print("-----------------")

    for mode in modes:
        test_i=list()
        eval_i=list()

        if mode == "roberta":
            dir_input = "baseline_n_result_open_domain"+str(datasets[dataset])+"_entropy_sscl"
            #print(dir_input)
        elif mode == "bert":
            dir_input = "baseline_bert_n_result_open_domain"+str(datasets[dataset])+"_entropy_sscl"
            #print(dir_input)

        #1~5
        for i in times_list: #1~5
            file_dir = dir_input+"_"+str(i)
            try:
                fileDir = os.listdir(file_dir)
                #find file
                test_file = [f for f in fileDir if "test_paper_results" in f and "txt_no_eval" not in f and supervised_data[0] in f][0]
                eval_file = [f for f in fileDir if "eval_results" in f and supervised_data[0] in f][0]
                #test
                with open(file_dir+"/"+test_file) as f:
                    for line in f:
                        line = line.strip().split()
                        if line[0] == "F1":
                            test_i.append(float(line[-1].replace("%","")))
                #eval
                with open(file_dir+"/"+eval_file) as f:
                    for line in f:
                        line = line.strip().split()
                        if line[0] == "eval_accuracy":
                            eval_i.append(float(line[-1])*100)
                print("File:", file_dir, ": DONE")
            except:
                print("File:", file_dir, ": PENDING...")
                #print("No file")
                #print("Less than 5 times")
                pass

        ####Caculate
        print("Mode:",mode)
        try:
            mean_test_i = mean(test_i)
            mean_eval_i = mean(eval_i)
            std_test_i = pstdev(test_i)
            std_eval_i = pstdev(eval_i)
            print("test:","{:.2f}%".format(mean_test_i)," ; ","std:","{:.2f}".format(std_test_i))
            print("eval:","{:.2f}%".format(mean_eval_i)," ; ","std:","{:.2f}".format(std_eval_i))
            print("test:",test_i)
            print("eval:",eval_i)
            print("------")
        except:
            print("test:","Pending...")
            print("eval:","Pending...")
            print("------")
    print("-------------------------------------------")
    ###################################
    ###################################




'''
    #st
    ###################################
    ###################################
    #print("Dataset:",dataset)
    #print("!!!!!!!!!!!!!!!!!")
    print("#################")
    print("Model:","st")
    print("#################")
    print("-----------------")

    for mode in modes:

        if mode == "roberta":
            dir_input = "retriver_n_result_open_domain"+str(datasets[dataset])+"_entropy_st"
            #print(dir_input)
        elif mode == "bert":
            dir_input = "retriver_bert_n_result_open_domain"+str(datasets[dataset])+"_entropy_st"
            #print(dir_input)

        #1~5
        for k in ["16","32","48"]:
            test_i=list()
            eval_i=list()
            for i in times_list: #1~5
                file_dir = dir_input+"_"+str(k)+"_"+str(i)
                try:
                    fileDir = os.listdir(file_dir)
                    #find file
                    test_file = [f for f in fileDir if "test_paper_results" in f and "txt_no_eval" not in f and supervised_data[0] in f][0]
                    eval_file = [f for f in fileDir if "eval_results" in f and supervised_data[0] in f][0]
                    #test
                    with open(file_dir+"/"+test_file) as f:
                        for line in f:
                            line = line.strip().split()
                            if line[0] == "F1":
                                test_i.append(float(line[-1].replace("%","")))
                    #eval
                    with open(file_dir+"/"+eval_file) as f:
                        for line in f:
                            line = line.strip().split()
                            if line[0] == "eval_accuracy":
                                eval_i.append(float(line[-1])*100)
                    print("File:", file_dir, ": DONE")
                except:
                    print("File:", file_dir, ": PENDING...")
                    #print("Less than 5 times")
                    pass

            ####Caculate
            print("Mode:",mode)
            print("K:",k)
            try:
                mean_test_i = mean(test_i)
                mean_eval_i = mean(eval_i)
                std_test_i = pstdev(test_i)
                std_eval_i = pstdev(eval_i)
                print("test:","{:.2f}%".format(mean_test_i)," ; ","std:","{:.2f}".format(std_test_i))
                print("eval:","{:.2f}%".format(mean_eval_i)," ; ","std:","{:.2f}".format(std_eval_i))
                print("test:",test_i)
                print("eval:",eval_i)
                print("------")
            except:
                print("test:","Pending...")
                print("eval:","Pending...")
                print("------")
        print("-------------------------------------------")
    print("===========================================")
    ###################################
    ###################################


    print("===========================================")
    print("===========================================")



    #sscl_dt
    ###################################
    ###################################
    #print("Dataset:",dataset)
    #print("!!!!!!!!!!!!!!!!!")
    print("#################")
    print("Model:","sscl_dt")
    print("#################")
    print("-----------------")

    for mode in modes:

        if mode == "roberta":
            dir_input = "retriver_n_result_open_domain"+str(datasets[dataset])+"_entropy_sscl_dt"
            #print(dir_input)
        elif mode == "bert":
            dir_input = "retriver_bert_n_result_open_domain"+str(datasets[dataset])+"_entropy_sscl_dt"
            #print(dir_input)

        #1~5
        for k in ["16","32","48"]:
            test_i=list()
            eval_i=list()
            for i in times_list: #1~5
                file_dir = dir_input+"_"+str(k)+"_"+str(i)
                try:
                    fileDir = os.listdir(file_dir)
                    #find file
                    test_file = [f for f in fileDir if "test_paper_results" in f and "txt_no_eval" not in f and supervised_data[0] in f][0]
                    eval_file = [f for f in fileDir if "eval_results" in f and supervised_data[0] in f][0]
                    #test
                    with open(file_dir+"/"+test_file) as f:
                        for line in f:
                            line = line.strip().split()
                            if line[0] == "F1":
                                test_i.append(float(line[-1].replace("%","")))
                    #eval
                    with open(file_dir+"/"+eval_file) as f:
                        for line in f:
                            line = line.strip().split()
                            if line[0] == "eval_accuracy":
                                eval_i.append(float(line[-1])*100)
                    print("File:", file_dir, ": DONE")
                except:
                    print("File:", file_dir, ": PENDING...")
                    #print("Less than 5 times")
                    pass

            ####Caculate
            print("Mode:",mode)
            print("K:",k)
            try:
                mean_test_i = mean(test_i)
                mean_eval_i = mean(eval_i)
                std_test_i = pstdev(test_i)
                std_eval_i = pstdev(eval_i)
                print("test:","{:.2f}%".format(mean_test_i)," ; ","std:","{:.2f}".format(std_test_i))
                print("eval:","{:.2f}%".format(mean_eval_i)," ; ","std:","{:.2f}".format(std_eval_i))
                print("test:",test_i)
                print("eval:",eval_i)
                print("------")
            except:
                print("test:","Pending...")
                print("eval:","Pending...")
                print("------")
        print("-------------------------------------------")
    print("===========================================")
    ###################################
    ###################################


'''
