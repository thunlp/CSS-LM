import json

#{"sentence": "i would like to use a different operating system altogether.", "aspect": "operating system", "sentiment": "neutral"}


with open("../GCAE/atsa-laptop/atsa_test.json") as f:
    laptop_data = json.load(f)
laptop_num_label_dict = dict()
for l in laptop_data:
    if l["sentiment"] == "conflict":
        continue
    try:
        data = {"sentence": l["sentence"],"aspect":l["aspect"], "sentiment":l["sentiment"], "domain":"laptop"}
        laptop_num_label_dict[l["sentiment"]].append(data)
    except:
        laptop_num_label_dict[l["sentiment"]]=[]
        data = {"sentence": l["sentence"],"aspect":l["aspect"], "sentiment":l["sentiment"], "domain":"laptop"}
        laptop_num_label_dict[l["sentiment"]].append(data)


min_num = 99999999
for label in set(laptop_num_label_dict.keys()):
    print(label,len(laptop_num_label_dict[label]))
    if len(laptop_num_label_dict[label]) < min_num:
        min_num = len(laptop_num_label_dict[label])
print("======")
for label in laptop_num_label_dict.keys():
    laptop_num_label_dict[label] = laptop_num_label_dict[label][:min_num]
for label in laptop_num_label_dict.keys():
    print(label,len(laptop_num_label_dict[label]))
print("======")


with open("../GCAE/acsa-restaurant-large/acsa_test.json") as f:
    restaurant_data = json.load(f)
restaurant_num_label_dict = dict()
for l in restaurant_data:
    try:
        data = {"sentence": l["sentence"],"aspect":l["aspect"], "sentiment":l["sentiment"], "domain":"restaurant"}
        restaurant_num_label_dict[l["sentiment"]].append(data)
    except:
        restaurant_num_label_dict[l["sentiment"]]=[]
        data = {"sentence": l["sentence"],"aspect":l["aspect"], "sentiment":l["sentiment"], "domain":"restaurant"}
        restaurant_num_label_dict[l["sentiment"]].append(data)

for label in set(restaurant_num_label_dict.keys()):
    print(label,len(restaurant_num_label_dict[label]))
print("======")
for label in restaurant_num_label_dict.keys():
    restaurant_num_label_dict[label] = restaurant_num_label_dict[label][:min_num]
for label in restaurant_num_label_dict.keys():
    print(label,len(restaurant_num_label_dict[label]))
print("======")

total_dict = dict()
for label in laptop_num_label_dict.keys():
    total_dict[label] = restaurant_num_label_dict[label] + laptop_num_label_dict[label]
    print(len(total_dict[label]))
    print(label)
    print("---")

total_list = list()
for label in total_dict.keys():
    print(label)
    total_list += total_dict[label]

print(128*6, len(total_list))

with open("test.json","w") as f:
    json.dump(total_list,f)




