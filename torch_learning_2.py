import torch
import torch.nn as tnn
import torch.nn.functional as F
"""
两者功能相近。
torch.nn  是一个类，继承于nn.model。包含Containers（容器），非线性激活函数，卷积层，池化层，归一化层，线性层， dropout层，稀疏层，视觉层（Vision Layers），距离函数(distance function),
损失函数， 填充函数
torch.nn.functional 是一个函数,包含非线性激活函数，卷积函数，池化函数，归一化函数，线性函数， dropout函数，距离函数(distance function),
损失函数， 填充函数
"""

"""
你可能会疑惑为什么需要这两个功能如此相近的模块，其实这么设计是有其原因的。如果我们只保留nn.functional下的函数的话，
在训练或者使用时，我们就要手动去维护weight, bias, stride这些中间量的值，这显然是给用户带来了不便。而如果我们只保留nn下的类的话，
其实就牺牲了一部分灵活性，因为做一些简单的计算都需要创造一个类，这也与PyTorch的风格不符。

但PyTorch官方推荐：具有学习参数的（例如，conv2d, linear, batch_norm)采用nn.xxx方式，
没有学习参数的（例如，maxpool, loss func, activation func）等根据个人选择使用nn.functional.xxx或者nn.Xxx方式。
但关于dropout，个人强烈推荐使用nn.Xxx方式，因为一般情况下只有训练阶段才进行dropout，在eval阶段都不会进行dropout。

torch.nn.Module 所有网络的基类。 你的模型也应该继承这个类。
"""

"""
class Conv1d(_ConvNd): #nn的卷积
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
                

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1): #function的卷积
    if input is not None and input.dim() != 3:
        raise ValueError("Expected 3D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = ConvNd(_single(stride), _single(padding), _single(dilation), False,
               _single(0), groups, torch.backends.cudnn.benchmark,
               torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
    return f(input, weight, bias)
"""



x = torch.ones((2,3), requires_grad=True)
print(x)

"""
https://pytorch-cn.readthedocs.io/zh/latest/package_references/functional/#torchnnfunctional
https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#torchnn

https://www.cnblogs.com/silence-cho/p/11404817.html
"""