import json

label_type = list()
train_list = list()
with open("org/train_5500.txt",encoding="ISO-8859-1") as f:
    for line in f:
        train_dict = dict()
        #print(line)
        line = line.strip().split(":")
        label_type.append(line[0])
        train_list.append({"sentence":line[1], "aspect":"QA", "sentiment":line[0]})
print(len(set(label_type)))
with open("train_all.json","w") as f:
    json.dump(train_list,f)
with open("dev.json","w") as f:
    json.dump(train_list,f)


#label_type = list()
test_list = list()
with open("org/test_10.txt",encoding="ISO-8859-1") as f:
    for line in f:
        text_dict = dict()
        #print(line)
        line = line.strip().split(":")
        #label_type.append(line[0])
        test_list.append({"sentence":line[1], "aspect":"QA", "sentiment":line[0]})
with open("test.json","w") as f:
    json.dump(test_list,f)
