import json
import sys
import random

file = sys.argv[1]
fewshot_n = int(sys.argv[2])
with open(file) as f:
    data = json.load(f)

label_list = list()
for line in data:
    label_list.append(line["sentiment"])

total_len = len(label_list)
label_list = list(set(label_list))

label_dict = dict()
for line in data:
    try:
        label_dict[line["sentiment"]].append(line)
    except:
        label_dict[line["sentiment"]] = []
        label_dict[line["sentiment"]].append(line)

#print("Number:",fewshot_n*len(label_list))
print("=========")
print("Sample N:",fewshot_n)
print("Total len",total_len)
print("=========")

train_n = list()
for label in label_list:
    print("+++",len(label_dict[label]))
    #train_n += random.choices(label_dict[label],k=fewshot_n)
    #samples = random.sample(label_dict[label],min(len(label_dict[label]),fewshot_n))
    ratio_num =round( fewshot_n*(len(label_dict[label])/total_len))+1
    print("==",ratio_num)
    samples = random.sample(label_dict[label],min(len(label_dict[label]),ratio_num))
    if len(samples) < ratio_num:
        samples += random.choices(label_dict[label],k=ratio_num-len(samples))
    train_n += samples
    #print(train_n)
    print(label,len(samples))
    print("--------")

print("=========")
print("Final Sample",len(train_n))

#with open("train_"+str(fewshot_n)+".json", 'w') as f:
with open("train.json_"+str(fewshot_n), 'w') as f:
    json.dump(train_n, f)
