import json

s=list()
with open("train.json","r") as f:
    f = json.load(f)
    for line in f:
        s.append(line["sentiment"])
print(sorted(list(set(s))))
print(len(sorted(list(set(s)))))



s=list()
with open("test.json","r") as f:
    f = json.load(f)
    for line in f:
        s.append(line["sentiment"])
print(sorted(list(set(s))))
print(len(sorted(list(set(s)))))
