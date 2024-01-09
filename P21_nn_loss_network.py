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
for data in dataloader:
    imgs,targets=data
    outputs=zna(imgs)

    # 损失函数：您使用了nn.CrossEntropyLoss作为损失函数，这是适用于多类别分类问题的常用损失函数。
    result_loss=loss(outputs,targets)
    # print(result_loss)# tensor(2.4262, grad_fn=<NllLossBackward0>)
    result_loss.backward()
    print("ok")

    # print(outputs)
    # print(targets)
# tensor([[ 0.0817, -0.0125, -0.0615, -0.1308, -0.0702,  0.0358,  0.0134,  0.1079,
#          -0.0905,  0.0910]], grad_fn=<AddmmBackward0>)
# tensor([3])

# writer=SummaryWriter("logs")
# writer.add_graph(zna,input)
# writer.close()


