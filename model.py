# 搭建神经网络
from torch import nn
import torch


class Zna(nn.Module):
    def __init__(self):
        super(Zna, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    zna=Zna()
    input=torch.ones((64,3,32,32))
    output=zna(input)
    print(output.shape)# torch.Size([64, 10])