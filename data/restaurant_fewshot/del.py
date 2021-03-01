import json

with open("/data1/private/suyusheng/task_selecte/data/sciie_fewshot/train_all.json","r") as f:
    f = json.load(f)
    print("Train:",len(f))

with open("/data1/private/suyusheng/task_selecte/data/sciie_fewshot/test.json","r") as f:
    f = json.load(f)
    print("Test:",len(f))
