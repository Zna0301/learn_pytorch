import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

# 使用PyTorch为CIFAR-10数据集创建一个简单的卷积神经网络（CNN）。你的网络命名为Zna，目前包含一个卷积层（Conv2d）

# 数据集
# 数据集和数据加载器：使用torchvision加载了CIFAR-10数据集，并创建了一个DataLoader以处理训练或评估过程中的数据批次。
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64)

# Zna类：这是你的神经网络类。
# 它继承自nn.Module，这是PyTorch中所有神经网络模块的基类。在类内部，你在__init__方法中定义了一个卷积层（Conv2d）。
class Zna(nn.Module):
    def __init__(self):
        super(Zna, self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    # 前向方法：你实现了前向方法，定义了数据在网络中的流动方式。在这种情况下，它通过卷积层执行前向传播。
    def forward(self,x):
        x=self.conv1(x)
        return x

# 实例化模型：你创建了Zna类的一个实例，并打印了模型的架构。
zna=Zna()
print(zna)
# Zna(
#   (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
# )

writer=SummaryWriter("logs")
step=0
# 通过dataloader迭代获取数据批次，将它们传递给你的Zna模型，并打印输出的形状。这是在神经网络的训练或评估阶段中常见的模式。
for data in dataloader:
    imgs,targets=data
    output=zna(imgs)
    print(imgs.shape) # torch.Size([64, 3, 32, 32])
    print(output.shape) # torch.Size([64, 6, 30, 30])->[x,3,30,30]

    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("input",imgs,step)
    writer.add_images("output",output,step)
    step=step+1

    writer.close()


