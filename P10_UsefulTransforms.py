from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 通过Image.open方法读取了一张图像，路径为"images/img1.png"
img=Image.open("images/img.webp")
# 使用print函数打印了这个img对象，这将显示图像的一些基本信息，如图像的尺寸、模式等。
print(img)# <PIL.WebPImagePlugin.

# 1.ToTensor的使用
trans_totensor=transforms.ToTensor()
# 通过调用trans_totensor的__call__方法，将img对象作为参数传入，将图像转换为张量格式，并将结果保存在img_tensor变量中。
img_tensor=trans_totensor(img)

# 使用writer.add_image方法将转换后的图像张量img_tensor添加到TensorBoard的日志中，指定了一个名称"Totensor"作为标识。
writer=SummaryWriter("logs")
writer.add_image("Totensor",img_tensor)


# 2.Normalize(归一化)的使用 input为ToTensor
# 计算方式output[channel] = (input[channel] - mean[channel]) / std[channel]``
# input[0,1] result[-1,1]
trans_norm=transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# Normalize方法需要在ToTensor方法之后使用，即img_tensor
img_norm=trans_norm(img_tensor)

# 归一化前后对比
print(img_tensor[0][0][0]) # tensor(0.4980)
print(img_norm[0][0][0]) # tensor(-0.0039)
writer.add_image("Normalize",img_norm,2)


# 3.Resize的使用 输入为<PIL.WebPImagePlugin.
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img) # Resize仍然返回是PIL类似
# 在transboard显示要变成ToTensor类型
img_resize=trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)


# 4.Compose 组合各种图像变换方式
# PIL->PIL->Tensor->Tensor
trans_compose=transforms.Compose([
    transforms.Resize(512),
    trans_totensor,
    trans_norm
])
# 调用transform的__call__方法，将img对象作为参数传入，依次应用变换序列，并保存在img_compose变量中
img_compose = trans_compose(img)
writer.add_image("Compose",img_compose,1)


# 5.RandomCrop（随机裁剪）input PIL
trans_random=transforms.RandomCrop(500)
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)


writer.close()