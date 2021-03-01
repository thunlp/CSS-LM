import csv

counter = 0
#max_num_sent = 1000000
max_num_sent = 500000

all_data_list = list()
with open("domain_data/all_fewshot.txt") as f:
    for line in f:
        if line == "\n":
            continue
        if counter < max_num_sent:
            counter+=1
            line = line.strip()
            #print(line)
            all_data_list.append([line])
        else:
            break

print("Domain:",counter)

with open("openwebtext.txt") as f:
    for line in f:
        if line == "\n":
            continue
        if counter < max_num_sent:
            counter+=1
            line = line.strip()
            all_data_list.append([line])
        else:
            break
print("All:",counter)

with open("train.txt", "w") as f:
    writer = csv.writer(f)
    writer.writerows(all_data_list)

