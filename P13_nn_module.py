import torch
from torch import nn

# 定义了一个简单的神经网络模型 ZnaNN，它只包含一个 forward 方法，该方法将输入加上1并返回结果
class ZnaNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output=input+1
        return output

# 创建了一个模型实例 zna，将输入张量 x 传递给模型进行前向传播
zna=ZnaNN()
x=torch.tensor(1.0)
output=zna(x)
print(output)