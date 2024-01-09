import torch
import torchvision

# 演示了如何使用 PyTorch 保存 VGG16 模型的两种不同方式：

vgg16=torchvision.models.vgg16(weights=None)

# 模型保存方式 1(保存模型结构及参数)
# 这种方式将整个模型的结构和参数保存到文件中。在加载时，可以通过torch.load加载整个模型。
torch.save(vgg16,"vgg16_method1.pdh")

# 方式 2(只保存网络模型的参数->字典形式)->推荐
# 这种方式只保存模型的参数，而不包括模型的结构。这是推荐的方式，因为它更轻量，而且在实际使用中更常见。
# 在加载时，需要首先创建相同结构的模型，然后使用 load_state_dict 方法加载保存的参数。
torch.save(vgg16.state_dict(),"vgg16_method2.pdh")


