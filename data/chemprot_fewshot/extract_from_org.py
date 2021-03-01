import json

#{"label": "main subject", "tokens": "For the 1971 film \" A Blank on the Map \" , he joined the first Western expedition to a remote highland valley in New Guinea to seek out a lost tribe .", "h": ["A Blank on the Map", ["Q4655508", 20, 38, 0.5]], "t": ["lost tribe", ["Q672979", 138, 148, 0.5]]}

train_label = list()
train_list = list()
train_label_dict_num = dict()
with open("org/train.txt","r") as f:
    for line in f:
        train_dict = dict()
        line = json.loads(line)
        if line["label"] == 'AGONIST-INHIBITOR' or line["label"] == 'SUBSTRATE_PRODUCT-OF':
            continue
        train_dict["sentiment"] = line["label"]
        train_label.append(line["label"])
        if line["label"] not in train_label_dict_num:
            train_label_dict_num[line["label"]] = 1
        else:
            train_label_dict_num[line["label"]] += 1
        train_dict["sentence"] = line["text"]
        train_dict["aspect"] = "chemprot"
        #h_site = line["metadata"][:2]
        #t_site = line["metadata"][2:]
        #line = line["text"].replace("[[","").replace("]]","").replace("<<","").replace(">>","")
        #line = line.strip().split()
        #h = " ".join(line[h_site[0]:h_site[1]+1])
        #t = " ".join(line[t_site[0]:t_site[1]+1])
        #train_dict["h"] = [h]
        #train_dict["t"] = [t]
        train_list.append(train_dict)


dev_label = list()
dev_list = list()
dev_label_dict_num = dict()
with open("org/dev.txt","r") as f:
    for line in f:
        dev_dict = dict()
        line = json.loads(line)
        if line["label"] == 'AGONIST-INHIBITOR' or line["label"] == 'SUBSTRATE_PRODUCT-OF':
            continue
        dev_dict["sentiment"] = line["label"]
        dev_label.append(line["label"])
        dev_dict["sentence"] = line["text"]
        dev_dict["aspect"] = "chemprot"
        #h_site = line["metadata"][:2]
        #t_site = line["metadata"][2:]
        #line = line["text"].replace("[[","").replace("]]","").replace("<<","").replace(">>","")
        #line = line.strip().split()
        #h = " ".join(line[h_site[0]:h_site[1]+1])
        #t = " ".join(line[t_site[0]:t_site[1]+1])
        #dev_dict["h"] = [h]
        #dev_dict["t"] = [t]
        if line["label"] not in dev_label_dict_num:
            dev_label_dict_num[line["label"]] = 1
        else:
            dev_label_dict_num[line["label"]] += 1
        dev_list.append(dev_dict)

test_label = list()
test_list = list()
test_label_dict_num = dict()
with open("org/test.txt","r") as f:
    for line in f:
        test_dict = dict()
        line = json.loads(line)
        if line["label"] == 'AGONIST-INHIBITOR' or line["label"] == 'SUBSTRATE_PRODUCT-OF':
            continue
        test_dict["sentiment"] = line["label"]
        test_label.append(line["label"])
        test_dict["sentence"] = line["text"]
        test_dict["aspect"] = "chemprot"
        #h_site = line["metadata"][:2]
        #t_site = line["metadata"][2:]
        #line = line["text"].replace("[[","").replace("]]","").replace("<<","").replace(">>","")
        #line = line.strip().split()
        #h = " ".join(line[h_site[0]:h_site[1]+1])
        #t = " ".join(line[t_site[0]:t_site[1]+1])
        #test_dict["h"] = [h]
        #test_dict["t"] = [t]
        if line["label"] not in test_label_dict_num:
            test_label_dict_num[line["label"]] = 1
        else:
            test_label_dict_num[line["label"]] += 1
        test_list.append(test_dict)

print(len(set(train_label)))
print(len(set(dev_label)))
print(len(set(test_label)))
print("========")
'''
print(sorted(list(set(train_label_dict_num.keys()))))
print(sorted(list(set(dev_label_dict_num.keys()))))
print(sorted(list(set(test_label_dict_num.keys()))))
'''
for l in sorted(list(set(train_label_dict_num.keys()))):
    print(l)
print("=====")
for l in sorted(list(set(dev_label_dict_num.keys()))):
    print(l)
print("=====")
for l in sorted(list(set(test_label_dict_num.keys()))):
    print(l)


with open("train_all.json","w", encoding='utf-8') as f:
    json.dump(train_list,f)

with open("dev.json","w", encoding='utf-8') as f:
    json.dump(dev_list,f)

with open("test.json","w", encoding='utf-8') as f:
    json.dump(test_list,f)
