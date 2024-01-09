import torch
import torchvision

# 演示了两种不同的方式来加载保存的 VGG16 模型：

# 模型加载方式 1 加载整个模型结构及参数
model=torch.load("vgg16_method1.pdh")
# print(model)

# 加载方式 2 加载模型参数的字典形式：
vgg16=torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pdh"))
print(vgg16)