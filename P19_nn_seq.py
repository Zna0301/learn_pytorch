import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


# 定义了一个使用 PyTorch 的简单卷积神经网络（CNN）模型。这个模型名为 Zna，包括三个卷积层，后面跟着最大池化层，然后是两个全连接（线性）层。
# 这个模型设计用于图像分类任务，其中输入具有3个通道（例如，RGB图像），输出是每个类别的分类分数。
# 卷积层后面跟着最大池化层，以降采样空间维度，全连接层执行基于展平特征的最终分类。
from torch.utils.tensorboard import SummaryWriter


class Zna(nn.Module):
    def __init__(self):
        super(Zna, self).__init__()
        # 第一卷积层，输出通道数为32，卷积核大小为5，填充为2
        # self.conv1=Conv2d(3,32,5,padding=2)
        # # 第一个最大池化层，池化核大小为2
        # self.maxpool1=MaxPool2d(2)
        # self.conv2=Conv2d(32,32,5,padding=2)
        # self.maxpool2=MaxPool2d(2)
        # self.conv3=Conv2d(32,64,5,padding=2)
        # self.maxpool3=MaxPool2d(2)
        # # 展平层，将4D张量转换为2D以传递给全连接层
        # self.flatten=Flatten()
        # # 第一个全连接层，输入特征数为1024，输出特征数为64
        # self.linear1=Linear(1024,64)
        # self.linear2=Linear(64,10)

        self.model1=Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64,10)
        )

    def forward(self,x):
        # 通过各层的前向传播
        # x=self.conv1(x)
        # x=self.maxpool1(x)
        # x=self.conv2(x)
        # x=self.maxpool2(x)
        # x=self.conv3(x)
        # x=self.maxpool3(x)
        # x=self.flatten(x)
        # x=self.linear1(x)
        # x=self.linear2(x)
        x=self.model1(x)
        return x

zna=Zna()
print(zna)
# Zna(
#   (conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (conv3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   (maxpool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear1): Linear(in_features=1024, out_features=64, bias=True)
#   (linear2): Linear(in_features=64, out_features=10, bias=True)
# )
#

input=torch.ones((64,3,32,32))
output=zna(input)
print(output.shape)# torch.Size([64, 10])

writer=SummaryWriter("logs")
writer.add_graph(zna,input)
writer.close()
