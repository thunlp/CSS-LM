import json
import csv

restaurant_fewshot=dict()
sciie_fewshot=dict()
sst2_fewshot=dict()
trec_fewshot=dict()


with open("../../restaurant_fewshot/train_all.json") as f:
    restaurant_fewshot = json.load(f)
restaurant_fewshot_list=list()
sentence_old="abc"
for line in restaurant_fewshot:
    if line["sentence"] != sentence_old:
        sentence_old = line["sentence"]
        restaurant_fewshot_list.append([line["sentence"]])
    else:
        continue


with open("../../sciie_fewshot/train_all.json") as f:
    sciie_fewshot = json.load(f)
sciie_fewshot_list=list()
sentence_old="abc"
for line in sciie_fewshot:
    if line["sentence"] != sentence_old:
        sentence_old = line["sentence"]
        sciie_fewshot_list.append([line["sentence"]])
    else:
        continue


with open("../../sst2_fewshot/train_all.json") as f:
    sst2_fewshot = json.load(f)
sst2_fewshot_list=list()
sentence_old="abc"
for line in sst2_fewshot:
    if line["sentence"] != sentence_old:
        sentence_old = line["sentence"]
        sst2_fewshot_list.append([line["sentence"]])
    else:
        continue


with open("../../trec_fewshot/train_all.json") as f:
    trec_fewshot = json.load(f)
trec_fewshot_list=list()
sentence_old="abc"
for line in trec_fewshot:
    if line["sentence"] != sentence_old:
        sentence_old = line["sentence"]
        trec_fewshot_list.append([line["sentence"]])
    else:
        continue

###
with open("../../scicite_fewshot/train_all.json") as f:
    scicite_fewshot = json.load(f)
scicite_fewshot_list=list()
sentence_old="abc"
for line in scicite_fewshot:
    if line["sentence"] != sentence_old:
        sentence_old = line["sentence"]
        scicite_fewshot_list.append([line["sentence"]])
    else:
        continue

with open("../../chemprot_fewshot/train_all.json") as f:
    chemprot_fewshot = json.load(f)
chemprot_fewshot_list=list()
sentence_old="abc"
for line in chemprot_fewshot:
    if line["sentence"] != sentence_old:
        sentence_old = line["sentence"]
        chemprot_fewshot_list.append([line["sentence"]])
    else:
        continue


with open("../../sst5_fewshot/train_all.json") as f:
    sst5_fewshot = json.load(f)
sst5_fewshot_list=list()
sentence_old="abc"
for line in sst5_fewshot:
    if line["sentence"] != sentence_old:
        sentence_old = line["sentence"]
        sst5_fewshot_list.append([line["sentence"]])
    else:
        continue

with open("../../aclintent_fewshot/train_all.json") as f:
    aclintent_fewshot = json.load(f)
aclintent_fewshot_list=list()
sentence_old="abc"
for line in aclintent_fewshot:
    if line["sentence"] != sentence_old:
        sentence_old = line["sentence"]
        aclintent_fewshot_list.append([line["sentence"]])
    else:
        continue
###

print(len(restaurant_fewshot_list))
print(len(sciie_fewshot_list))
print(len(sst2_fewshot_list))
print(len(trec_fewshot_list))
###
print(len(scicite_fewshot_list))
print(len(chemprot_fewshot_list))
print(len(sst5_fewshot_list))
print(len(aclintent_fewshot_list))
###

with open("restaurant_fewshot.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(restaurant_fewshot_list)

with open("sciie_fewshot.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(sciie_fewshot_list)

with open("sst2_fewshot.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(sst2_fewshot_list)

with open("trec_fewshot.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(trec_fewshot_list)

###
with open("scicite_fewshot.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(scicite_fewshot_list)

with open("chemprot_fewshot.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(chemprot_fewshot_list)

with open("sst5_fewshot.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(sst5_fewshot_list)

with open("aclintent_fewshot.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(aclintent_fewshot_list)
###

#all_fewshot_list = restaurant_fewshot_list + sciie_fewshot_list + sst2_fewshot_list + trec_fewshot_list + scicite_fewshot_list + chemprot_fewshot_list + sst5_fewshot_list + aclintent_fewshot_list
all_fewshot_list = restaurant_fewshot_list + sciie_fewshot_list + scicite_fewshot_list + chemprot_fewshot_list + sst5_fewshot_list + aclintent_fewshot_list + sst2_fewshot_list
with open("all_fewshot.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(all_fewshot_list)
