import torch
import time

x = torch.rand(25128895,768)
y = torch.rand(1,768)

start = time.ctime()

m = y.matmul(x.T)


end = time.ctime()

print(start-end)

