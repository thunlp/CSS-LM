import json
import csv

with open("train.json") as f:
    data = json.load(f)

all_list = list()
for t in data:
    all_list.append([t["sentence"]])
    #all_list.append([])
#print(all_list)

#all_list = [[]]+[["restaurant"]]+[[]]+all_list

with open('train.txt', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(all_list)

