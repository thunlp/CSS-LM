import json
#{"sentence": "oh yeah...the view was good, too.", "aspect": "location", "sentiment": "positive"}

sentiment_map = {"0":"negative","1":"postive"}

train_list = list()
with open("org/train.tsv") as f:
    for line in f:
        line = line.strip().split("\t")
        train_list.append({"sentence":line[0], "aspect":"movie", "sentiment":sentiment_map[line[1]]})
with open("train.json","w") as f:
    json.dump(train_list,f)


dev_list = list()
with open("org/valid.tsv") as f:
    for line in f:
        line = line.strip().split("\t")
        dev_list.append({"sentence":line[0], "aspect":"movie", "sentiment":sentiment_map[line[1]]})
with open("dev.json","w") as f:
    json.dump(dev_list,f)


test_list = list()
with open("org/test.tsv") as f:
    for line in f:
        line = line.strip().split("\t")
        test_list.append({"sentence":line[0], "aspect":"movie", "sentiment":sentiment_map[line[1]]})
with open("test.json","w") as f:
    json.dump(test_list,f)
