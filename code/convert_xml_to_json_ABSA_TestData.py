import json
from xmljson import badgerfish as bf
from xml.dom import minidom


file_1="../data/laptop_restaurant_2014/test/ABSA_TestData_PhaseA/Laptops_Test_Data_PhaseA.xml"
file_2="../data/laptop_restaurant_2014/test/ABSA_TestData_PhaseA/Restaurants_Test_Data_PhaseA.xml"
mydoc_1 = minidom.parse(file_1)
texts_1 = mydoc_1.getElementsByTagName('text')
all_data_list_1 = list()
for text in texts_1:
    all_data_list_1.append({"sentence":text.firstChild.nodeValue,"aspect":"laptops", "sentiment":"laptops"})

mydoc_2 = minidom.parse(file_2)
texts_2 = mydoc_2.getElementsByTagName('text')
all_data_list_2 = list()
for text in texts_2:
    all_data_list_2.append({"sentence":text.firstChild.nodeValue,"aspect":"restaurants", "sentiment":"restaurants"})

all_data_list = all_data_list_1 + all_data_list_2

with open('../data/laptop_restaurant_2014/test/ABSA_TestData_PhaseA/lap_rest_test.json', 'w') as outfile:
    json.dump(all_data_list, outfile)
