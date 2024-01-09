import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=1)

from torch.utils.tensorboard import SummaryWriter

class Zna(nn.Module):
    def __init__(self):
        super(Zna, self).__init__()

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
        x=self.model1(x)
        return x

loss=nn.CrossEntropyLoss()
zna=Zna()

# 添加了优化器（使用SGD优化器）和训练循环来迭代训练模型 将模型参数zna.parameters()传递给优化器，并设置学习率为0.01
optim=torch.optim.SGD(zna.parameters(),lr=0.01)

# 您使用两个嵌套的循环来进行训练。外层循环是epoch循环，控制整个训练过程的迭代次数。内层循环是对数据加载器中的每个批次进行迭代。
for epoch in range(20):

    running_loss=0.0
    for data in dataloader:
        imgs,targets=data
        outputs=zna(imgs)

        # 损失函数：您使用了nn.CrossEntropyLoss作为损失函数，这是适用于多类别分类问题的常用损失函数。
        # 损失累加：在内层循环中，您定义了一个running_loss变量用于累加每个批次的损失值。
        result_loss=loss(outputs,targets)
        # print(result_loss)# tensor(2.4262, grad_fn=<NllLossBackward0>)

        # 优化器的使用：在每个批次中，您调用了optim.zero_grad()来清零梯度，
        # 然后使用result_loss.backward()进行反向传播计算梯度，最后使用optim.step()来更新模型参数。
        optim.zero_grad()
        result_loss.backward()
        optim.step()

        # 在每个批次中，您将当前批次的损失值累加到running_loss中。
        running_loss=running_loss+result_loss

    # 打印损失：在每个epoch结束后，您打印了running_loss的值。这里需要注意的是，由于您是累加损失值，所以打印出来的值可能会非常大。
    # 如果您希望得到每个epoch的平均损失值，可以将running_loss除以数据集的大小或批次数。
    print(running_loss)
    # tensor(18752.7969, grad_fn= < AddBackward0 >)
    # tensor(16235.8096, grad_fn= < AddBackward0 >)
    # tensor(15453.4443, grad_fn= < AddBackward0 >)



