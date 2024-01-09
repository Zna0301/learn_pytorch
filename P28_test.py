import torch
from PIL import Image
import torchvision
from torch import nn

# 您加载了一个预训练的模型，并将其移动到了GPU上进行推理。
# 您还将输入图像进行了相应的预处理，并将其移动到了GPU上。然后，您使用模型进行了推理，并打印了输出和预测结果。

image_path="./images/img.jpg"
image=Image.open(image_path)
print(image)
device=torch.device("cuda")

image=image.convert('RGB')
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                          torchvision.transforms.ToTensor(),])

image=transform(image)
print(image.shape)

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

# 加载网络模型
model=torch.load("zna_29_gpu.pth")
# 加载了一个名为"zna_0.pth"的模型，并将其移动到了GPU上
model=model.to(device)
print(model)

# 将输入图像进行了形状重塑，并将其移动到了GPU上。
image=torch.reshape(image,(1,3,32,32))
image=image.to(device)
# 将模型设置为评估模式（model.eval()）
model.eval()

# 禁用了梯度计算。
# 使用torch.no_grad()上下文管理器执行了推理，并打印了输出张量和预测结果。
with torch.no_grad():
    output=model(image)

# output是模型的输出张量，它包含了每个类别的预测分数。
# output.argmax(1)返回了预测结果的索引，即具有最高预测分数的类别的索引。
print(output)
print(output.argmax(1))