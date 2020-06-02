import torch

x = torch.Tensor(2, 3)
y = torch.Tensor(4, 2, 3)
# print(x+y)

x = torch.empty(5, 3)  #构造一个未初始化的5x3矩阵
print(x)

x = torch.rand(2, 3) #构建一个随机初始化矩阵
print(x)

x = torch.zeros(2, 4, dtype=torch.long)  #填充0的矩阵
print(x)

x = torch.Tensor([[2.0, 3.33]]) #直接构造张量
print(x)


y = x.new_ones(2, 2, dtype=torch.double)
print(y)

y = torch.randn_like(x, dtype=torch.float)
print(y)

print(y.size())

print(x+y)  #加法
print(torch.add(x, y))
y.add_(x)
print(y)

#调整大小
x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y)
z = x.view(-1, 8)
print(z)

#将torch张量转为python数字
x = torch.randn(1)
print(x)
print(x.item())

#二、Numpy与Torch之间转换
a = torch.ones(5)
print(a)
aa = a.numpy()
print(aa) #转为numpy列表

b = torch.ones(1,3)
print(b)
print(b.numpy())

a.add_(1) #说明a与aa访问同一块内存
a = torch.add(a, 1) #这样是不同的
print(a)
print(aa)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)  #numpy转torch
print(a)
print(b)
c = np.add(a, 1, out=a)
print(c)

#CUDA张量
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))