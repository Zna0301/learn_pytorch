import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# from model import *

# 定义训练设备
device=torch.device("cuda")
# device=torch.device("cpu")


# 训练数据集
train_data=torchvision.datasets.CIFAR10(root="./data",train=True,transform=torchvision.transforms.ToTensor(),
                                        download=True)
# 测试数据集
test_data=torchvision.datasets.CIFAR10(root="./data",train=False,transform=torchvision.transforms.ToTensor(),
                                        download=True)

# 计算数据集长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集长度为:{}".format(train_data_size))# 50000
print("测试数据集长度为:{}".format(test_data_size))# 10000

# 利用dataloader加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

# 搭建神经网络->在model.py
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

# 创建网络模型
zna=Zna()
zna=zna.to(device)

# 创建损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.to(device)

# 优化器
# learning_rate=0.01
learning_rate=1e-2 #->0.01
optimizer=torch.optim.SGD(zna.parameters(),lr=learning_rate)

# 设置训练网络的参数
# 记录训练次数
train_total_step=0
# 记录测试次数
test_total_step=0
# 训练轮数
epoch=30

# 添加tensorboard
writer=SummaryWriter("logs_train")

# 训练开始时间
start_time=time.time()

# 训练网络模型
for i in range(epoch):
    print("-----第{}轮训练开始-----".format((i+1)))

    # 训练步骤开始
    zna.train() # 可以不写 只对一部分网络层有作用
    for data in train_dataloader:

        imgs,targets=data
        imgs=imgs.to(device)
        targets=targets.to(device)
        outputs=zna(imgs)

        # 损失函数
        loss=loss_fn(outputs,targets)

        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total_step=train_total_step+1
        if train_total_step%100==0:
            end_time=time.time()
            print(end_time-start_time)
            print("训练次数为:{},Loss:{}".format(train_total_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),train_total_step)

    # 不调优 测试代码
    zna.eval() # 可以不写 只对一部分网络层有作用
    total_test_loss=0
    total_accaracy=0
    with torch.no_grad(): # 梯度为0 不用梯度来优化
        for data in test_dataloader:
            imgs,targets=data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs=zna(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            # 准确率->见笔记
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accaracy=total_accaracy+accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的准确率:{}".format(total_accaracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,test_total_step)
    writer.add_scalar("test_accaracy",total_accaracy/test_data_size,test_total_step)
    test_total_step+=1

    # 保存每一轮训练结果
    torch.save(zna,"zna_{}_gpu.pth".format(i))
    print("模型已经保存")

writer.close()









