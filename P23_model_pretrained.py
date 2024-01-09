import torchvision.datasets


from torch import nn

# 创建了一个 VGG16 模型的实例，但是没有使用任何预训练权重。
# weights=None 表示该模型将使用随机初始化的权重，而不会加载在特定数据集上预训练的权重。
vgg16_false=torchvision.models.vgg16(weights=None)
# 创建了一个 VGG16 模型的实例，并加载了在 ImageNet 数据集上预训练的权重
vgg16_true = torchvision.models.vgg16(weights='DEFAULT')


# print(vgg16_true)

train_data=torchvision.datasets.CIFAR10("./data",train=True,transform=torchvision.transforms.ToTensor(),download=True)

# 在加载预训练权重的 VGG16 模型上，你添加了一个新的线性层（nn.Linear(1000, 10)）作为分类器。
# 这是一种进行迁移学习或在不同任务上微调模型的常见方法，通过保持原始的卷积层权重，替换或添加新的全连接层来适应新的任务。
vgg16_true.add_module("add_linear",nn.Linear(1000,10))
# print(vgg16_true)# (add_linear): Linear(in_features=1000, out_features=10, bias=True)

vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
# print(vgg16_true)# (add_linear): Linear(in_features=1000, out_features=10, bias=True)

# 修改 添加
print(vgg16_false)
vgg16_false.classifier[6]=nn.Linear(4096,10)
print(vgg16_false)