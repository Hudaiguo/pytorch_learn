import torch
import numpy as np

"""pytorch的基本运算"""

x = torch.Tensor(2, 3)
y = torch.Tensor(4, 2, 3)
# print(x+y)

x = torch.empty(2, 3)  #构造一个未初始化的5x3矩阵
y = np.empty((2,3))    #构造一个未初始化的5x3矩阵
# print(x, y)

x = torch.zeros(2, 3, dtype=torch.long)  #填充0的矩阵
x1 = torch.ones(2,3)
y = np.zeros((2,3))
y1 = np.ones((2,3), dtype=np.uint8)
# print(x1, "\n", y1)

m= x.new_ones(2, 2)     #可继承原有属性
y = torch.randn_like(x, dtype=torch.float)
n = np.ones_like(y)     #返回一个用1填充的跟输入 形状和类型 一致的数组。
# print(m, n)

x = torch.rand(2, 3)       #构建一个随机初始化矩阵，[0, 1)的随机数
y = np.random.rand(2,3)    #构建一个随机初始化矩阵
z = np.random.randint(0, 10, (2,3), dtype=np.uint8)  #构建一个区间为[0-10]的随机初始化矩阵


x = torch.Tensor([[2.0, 3.33]])   #直接构造张量
x1 = torch.Tensor([2, 3])
y = np.array([[2.0, 3.33],
              [2.3, 3.54]])
y1 = np.array([2, 3])

#np与torch之间互转#np与torch之间互转
m = x.numpy()  #torch转numpy
n = torch.from_numpy(y) #numpy转torch
# print(m, "\n", n)
#将torch张量转为python数字
x = torch.randn(1)
m = x.item() #tensor转数字
# print(x, "\n", m)

# print(m.dtype)   #查看数据类型, torch及numpy都可以
print(type(m))   #torch与numpy都不可以，m只能为常量


#调整大小
x = torch.rand((4,4),)
m = x.view((-1, 8))
n = x.view((16))

#加法
x = torch.rand((2,3))
y = torch.ones((2,3))
m = x + y
n = torch.add(x, y)
l = y.add_(x)   #相当于 +=
# print(m, "\n", n, "\n", l)

#张量操作
x = torch.rand(5,3)
m = x[1:, 1]  #切片和索引
n = x[x>0.5]  #花式索引

print(x, "\n", m, "\n", n)