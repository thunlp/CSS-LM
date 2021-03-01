import json

s=list()
with open("test.txt","r") as f:
    for line in f:
        l = json.loads(line)["label"]
        #print(l)
        #print(l["label"])
        s.append(l)

print(len(s))
