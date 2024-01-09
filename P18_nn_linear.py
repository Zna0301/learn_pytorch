import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
import torch

dataset=torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64,drop_last=True)

class Zna(nn.Module):
    def __init__(self):
        super(Zna, self).__init__()
        self.linear1=Linear(196608,10)

    def forward(self,input):
        output=self.linear1(input)
        return output

zna=Zna()

for data in dataloader:
    imgs,targets=data
    print(imgs.shape)# torch.Size([64, 3, 32, 32])
    #output=torch.reshape(imgs,(1,1,1,-1))
    output=torch.flatten(imgs)
    print(output.shape)# torch.Size([1, 1, 1, 196608])->torch.Size([196608])

    output=zna(output)
    print(output.shape)# torch.Size([1, 1, 1, 10])->torch.Size([10])