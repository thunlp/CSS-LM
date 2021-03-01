import json
#{"sentence": "oh yeah...the view was good, too.", "aspect": "location", "sentiment": "positive"}

#sentiment_map = {"0":"negative","1":"postive"}

label_list = list()
train_list = list()
with open("org/train.txt") as f:
    for line in f:
        line = [l for l in line.strip().split("\t")]
        #print(line[0])
        #label = line[0]
        #sentence = line[1]
        label_list.append(line[0])
        train_list.append({"sentence":line[1], "aspect":"movie", "sentiment":line[0]})
with open("train_all.json","w") as f:
    json.dump(train_list,f)


dev_list = list()
with open("org/dev.txt") as f:
    for line in f:
        line = line.strip().split("\t")
        dev_list.append({"sentence":line[1], "aspect":"movie", "sentiment":line[0]})
with open("dev.json","w") as f:
    json.dump(dev_list,f)


test_list = list()
with open("org/test.txt") as f:
    for line in f:
        line = line.strip().split("\t")
        test_list.append({"sentence":line[1], "aspect":"movie", "sentiment":line[0]})
with open("test.json","w") as f:
    json.dump(test_list,f)


print(list(set(label_list)))
print(len(list(set(label_list))))
