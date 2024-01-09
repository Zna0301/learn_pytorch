import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,-0.5],
                    [-1,3]])

input=torch.reshape(input,(-1,1,2,2))
print(input.shape)# torch.Size([1, 1, 2, 2])

dataset=torchvision.datasets.CIFAR10("./data",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)


class Zna(nn.Module):
    def __init__(self):
        super(Zna, self).__init__()
        self.relu1=ReLU()
        self.sigmoid1=Sigmoid()

    def forward(self,input):
        # ReLU(x)=max(0,x)
        # output=self.relu1(input)
        output=self.sigmoid1(input)
        return output

zna=Zna()
# output=zna(input)
# print(output)
# # tensor([[[[1., 0.],
# #           [0., 3.]]]])

writer=SummaryWriter("logs")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,global_step=step)
    output=zna(imgs)
    writer.add_images("output",output,step)
    step+=1

writer.close()
