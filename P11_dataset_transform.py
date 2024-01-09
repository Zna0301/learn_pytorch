import torchvision
from torch.utils.tensorboard import SummaryWriter


dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# transform参数用于定义对图像进行的预处理操作，例如将图像转换为Tensor，并进行归一化。

# 使用torchvision.datasets.CIFAR10加载了CIFAR-10数据集，并将其分为训练集和测试集。
train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)

# print(test_set[0]) # 返回一个元组 (<PIL.Image.Image image mode=RGB size=32x32 at 0x1CD98FA6160>, 3)
# # 训练集中第一张图像是大小为32x32的RGB图像，表示为PIL库中的Image对象。输出中的(32, 32, 3)表示图像的尺寸为32x32，并且有3个通道（即RGB通道）。
#
# print((test_set.classes)) # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# # 返回一个包含CIFAR-10数据集类别标签的列表。在CIFAR-10数据集中，共有10个类别
#
# img,target=test_set[0]
# #img和target分别接收了test_set[0]的元组元素，其中img表示图像对象，target表示图像的标签。target的值为3，对应于类别标签'cat'。
#
# print(img)# (<PIL.Image.Image image mode=RGB size=32x32 at 0x28B529CD3D0>, 3)
# print(target) # target=3 --->'cat'
# print(test_set.classes[target])# cat
# # 将target的值作为索引，返回对应的类别标签。在这种情况下，它返回'cat'，表示该图像属于猫的类别。
# img.show()

#
# print(test_set[0]) # (tensor

# tensorboard使用
writer=SummaryWriter("logsP10")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)

writer.close()