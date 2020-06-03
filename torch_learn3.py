# -*- coding: utf-8 -*-
"""
@Time: 2020/6/3 22:20
@Author: Hudaiguo
@python version: 3.5.2
"""

import torch
import numpy as np

"""自动求导"""
"""
在pytorch搭建的神经网络中，Tensor 和Function为最主要的两个类，一起组成了一个无环图。 在前向传播时，Function操作tensor的值，
而进行反向传播时，需要计算function的导数来更新参数tensor， pytorch为我们自动实现了求导。每一个tensor都有一个requires_grad属性，
若tensor.requires_grad=True, 则无环图会记录对该tensor的所有操作，当进行backward时，pytorch就会自动计算其导数值，
并保存在tensor的grad属性中。
"""
x = torch.ones(2, 2, requires_grad=True)  #设置requires_grad=True， backward时会计算导数
y = x+2
    #属性值
        y.requirs_grad     是否autograd, 会自动继承x的requires_grad
        y.grad               导数或梯度值
        y.grad_fn           对x的操作function，grad_fn=<AddBackward0>
tensor.detach()           #将tensor从计算历史（无环图）中脱离出来？
with torch.no_grad():     #从计算历史（无环图）中脱离, backward时不求导
with torch.set_grad_enabled(phase == 'train')：  （phase == 'train'）为True时求导

tensor.backward()    #反向传播， 计算梯度，如果tensor只包含一个数时，backward不需要参数， 否则需要指明参数



#out为标量，所以backward时不带参数
x = torch.ones(2, 2, requires_grad=True)
y = x+2
z = y*y*3
out = z.mean()
out.backward()
print(x.grad)     #tensor([[4.5000, 4.5000],[4.5000, 4.5000]])
print(y.grad)     #None


#y不为为标量，backward时需要带参数
x = torch.ones(2, 2, requires_grad=True)
y = 2*x+2
y.backward(torch.tensor([[1,1],[1,1]], dtype=torch.float))  #可以理解为tensor([1, 1, 1, 1]) * dy/dx
print(x.grad)    # tensor([[2.,2.],[2.,2.]])


#y不为为标量，backward时需要带参数
x = torch.ones(2, 2, requires_grad=True)
y = 2*x+2
y.backward(torch.tensor([[1,0.1],[1,0.1]], dtype=torch.float))  #可以理解为tensor([1, 0.1, 1, 0.1]) * dy/dx
print(x.grad)    # tensor([[2.0000,0.2000],[2.0000,0.2000]])



#不明白这个的用处，反正自动求导