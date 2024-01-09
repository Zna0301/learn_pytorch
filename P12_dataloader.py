import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 测试集
# 创建了一个名为test_data的torchvision.datasets.CIFAR10对象，用于加载CIFAR-10测试集
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

# 使用test_data创建了一个测试集的数据加载器test_loader
# dataset: 数据加载器的数据集为test_data。batch_size: 每个批次的样本数量为4。
# shuffle: 在每个迭代中是否打乱数据，这里设置为True表示打乱数据。num_workers: 数据加载的并行工作进程数为0。
# drop_last: 如果数据集的大小不能被批次大小整除，是否丢弃最后一个不完整的批次，这里设置为False表示不丢弃。
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

# 测试集的第一张图像数据和标签 getitem() return img,target
img,target=test_data[0]
print(img.shape)# torch.Size([3, 32, 32])
print(target)# 3

writer=SummaryWriter("dataloader")
# 使用test_loader迭代加载测试集的数据。在每次迭代中，获取了一个批次的图像数据和标签，并打印了它们的形状和值。

for epoch in range(2):

    step=0
    for data in test_loader:
        imgs,targets=data
        # # batch_size=64时
        # print(imgs.shape) # torch.Size([4, 3, 32, 32])
        # # 每个批次的图像数据的形状为torch.Size([4, 3, 32, 32])，表示每个批次包含4张图像，
        # # 每张图像是一个3通道（RGB）的图像，大小为32x32像素。标签是一个大小为4的整数张量。
        # print(targets) # tensor([0, 5, 5, 2])
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step=step+1

writer.close()
