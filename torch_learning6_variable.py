# -*- coding: utf-8 -*-
"""
@Time:   2020/6/4 10:08
@Author: Hudaiguo
@python version: 3.5.2
"""
#注：tensor不能反向传播，variable可以反向传播。
#  在torch中的Variable就是一个存放会变化的值的地理位置。里面的值会不停发生变化，就像一个装鸡蛋的篮子，鸡蛋数会不断发生变化。那谁是里面的鸡蛋呢，自然就是torch的Tensor了

import torch
from torch.autograd import Variable  # torch 中 Variable 模块

tensor = torch.FloatTensor([[1, 2], [3, 4]])
# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensor, requires_grad=True)

print(tensor)
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable)
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""


"""
2.Variable计算以及梯度
对比tensor和variable的计算
"""
t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2
print(t_out)
print(v_out)    # 7.5


#到目前为止, 我们看不出什么不同, 但是时刻记住, Variable 计算时, 它在背景幕布后面一步步默默地搭建着一个庞大的系统, 叫做计算图, computational graph. 这个图是用来干嘛的? 原来是将所有的计算步骤 (节点) 都连接起来, 最后进行误差反向传递的时候, 一次性将所有 variable 里面的修改幅度 (梯度) 都计算出来, 而 tensor 就没有这个能力啦

v_out = torch.mean(variable*variable) #就是在计算图中添加的一个计算步骤, 计算误差反向传递的时候有他一份功劳, 我们就来举个例子:

v_out.backward()  # 模拟 v_out 的误差反向传递

# 下面两步看不懂没关系, 只要知道 Variable 是计算图的一部分, 可以用来传递误差就好.
# v_out = 1/4 * sum(variable*variable) 这是计算图中的 v_out 计算步骤
# 针对于 v_out 的梯度就是, d(v_out)/d(variable) = 1/4*2*variable = variable/2

print(variable.grad)  # 初始 Variable 的梯度
'''
 0.5000  1.0000
 1.5000  2.0000
'''


"""
3.获取Variable里面的数据
直接print(variable)只会输出 Variable 形式的数据, 在很多时候是用不了的(比如想要用 plt 画图), 所以我们要转换一下, 将它变成 tensor 形式.
"""

print(variable)  # Variable 形式
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data)  # tensor 形式
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data.numpy())  # numpy 形式
"""
[[ 1.  2.]
 [ 3.  4.]]
"""
