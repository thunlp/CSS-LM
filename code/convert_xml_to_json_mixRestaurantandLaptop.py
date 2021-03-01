import json
from xmljson import badgerfish as bf
from xml.dom import minidom


file="../data/laptops_Train.xml"
mydoc = minidom.parse(file)
#sentences = mydoc.getElementsByTagName('sentence')
texts = mydoc.getElementsByTagName('text')

all_data_list = list()
#for sentence in sentences:
for text in texts:
    #print(sentence.childNodes[0].data)
    all_data_list.append({"sentence":text.firstChild.nodeValue,"aspect":"mac","sentiment":"mac"})
    #print(text.firstChild.nodeValue)

with open("../data/restaurant/test.json") as f:
    id_dom = json.load(f)

all_data_list = id_dom+all_data_list

with open('../data/open_domain_mix/opendomain.json', 'w') as outfile:
    json.dump(all_data_list, outfile)
