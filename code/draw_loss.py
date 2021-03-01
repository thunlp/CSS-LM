import matplotlib.pyplot as plt
import os
import sys

dir = sys.argv[1]
fileDir = os.listdir(dir)
filename_test = {int(f.split("_")[-1].replace(".txt","")):f for f in fileDir if "test_paper_results_" in f and "best" not in f}
#x = sorted(list(filename_test.keys()))
#print(x)
#exit()
#filename_dev = {int(f.split("_")[-1].replace(".txt","")):f for f in fileDir if "test_results_" in f}
filename_dev = {int(f.split("_")[-1].replace(".txt","")):f for f in fileDir if "eval_results_" in f}
x = sorted(list(filename_dev.keys()))
#print(x)
#exit()


y_test_f1 = list()
y_dev_acc = list()
y_dev_loss = list()

for id_ in x:
    file_dev = filename_dev[id_]
    file_test = filename_test[id_]

    with open(dir+file_dev) as f:
        for line in f:
            line = line.strip().split()
            if line[0]=='eval_accuracy':
                y_dev_acc.append(float(line[-1]))
            elif line[0]=='eval_loss':
                y_dev_loss.append(float(line[-1]))
            else:
                pass

    with open(dir+file_test) as f:
        for line in f:
            line = line.strip().split()
            #print(line)
            if line[0]=='F1':
                y_test_f1.append(float(line[-1].replace("%",""))/100)
                #print(y_test_f1)
            else:
                pass


plt.plot(x, y_test_f1, label="test_f1", color='coral')
plt.scatter(x, y_test_f1, color='coral')

plt.plot(x, y_dev_loss, label="dev_loss", color='blue', linestyle='--')
plt.scatter(x, y_dev_loss, color='blue')

plt.plot(x, y_dev_acc, label="dev_f1", color='lightblue', linestyle='--')
plt.scatter(x, y_dev_acc, color='lightblue')

##########
###loss###
##########
loss_x = list()
loss_y = list()
loss_no_pseudo_x = list()
loss_no_pseudo_y = list()
with open(dir+"loss") as f:
    for i, line in enumerate(f):
        if i%x[0] == 0 and i>=x[0]:
            loss_x.append(i)
            loss_y.append(float(line))

try:
    with open(dir+"loss_no_pseudo") as f:
        for i, line in enumerate(f):
            if i%x[0] == 0 and i>=x[0]:
                loss_no_pseudo_x.append(i)
                loss_no_pseudo_y.append(float(line))
except:
    print("Have no loss_no_pseudo")

plt.plot(loss_x, loss_y, label="train_loss", color='lightgreen', linestyle='--')

try:
    plt.plot(loss_no_pseudo_x, loss_no_pseudo_y, label="train_loss_no_pseudo", color='green', linestyle='--')
except:
    pass

plt.legend(loc='upper left')
plt.savefig('loss.pdf',format="pdf")
