import json
import sys
import random

file = sys.argv[1]
fewshot_n = int(sys.argv[2])
with open(file) as f:
    data = json.load(f)

label_list = list()
for line in data:
    label_list.append(line["label"])

label_list = list(set(label_list))

label_dict = dict()
for line in data:
    try:
        label_dict[line["label"]].append(line)
    except:
        label_dict[line["label"]] = []
        label_dict[line["label"]].append(line)


print("Number:",fewshot_n*len(label_list))
full_flag = False
train = list()

train_n = list()
for label in label_list:
    #train_n += random.choices(label_dict[label],k=fewshot_n)
    samples = random.sample(label_dict[label],min(len(label_dict[label]),fewshot_n))
    if len(samples) < fewshot_n:
        samples += random.choices(label_dict[label],k=fewshot_n-len(samples))
    train_n += samples

print(len(train_n))
#with open("train_"+str(fewshot_n)+".json", 'w') as f:
with open("train.json_"+str(fewshot_n), 'w') as f:
    json.dump(train_n, f)



