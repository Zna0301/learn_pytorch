# SummaryWriter是PyTorch中用于将训练过程中的数据和结果写入TensorBoard可视化工具的类。
# TensorBoard是一个用于可视化和分析深度学习模型训练过程的强大工具，可以展示训练曲线、模型结构、图像、直方图等信息。
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image

# 创建实例 建了一个SummaryWriter实例，并指定了日志保存的目录路径
writer=SummaryWriter("logs")

image_path="data/train/bees_image/16838648_415acd9e3f.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)

print(type(img_array)) # <class 'numpy.ndarray'>
print(img_array.shape) # (512, 768, 3)

# add_image()方法2是SummaryWriter类提供的另一个方法，用于将图像数据写入TensorBoard。
# img_tensor的形状默认为(3, H, W) 其中H和W分别表示图像的高度和宽度。
# img_tensor也可以具有以下形状之一：(1, H, W)、(H, W)或(H, W, 3)。
# 只要相应的dataformats参数被传递，这些形状也是合适的。例如，可以使用CHW、HWC或HW作为dataformats参数。
writer.add_image("train",img_array,2,dataformats='HWC')


# 方法1
# 使用add_scalar方法将标量数据写入TensorBoard。调用add_scalar方法来添加名为"y=2x"的标量数据。
# 第一个参数是标量数据的名称，第二个参数是要记录的标量值，第三个参数是当前的步数。
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

# 使用writer.close()方法关闭SummaryWriter实例，确保所有数据都被写入TensorBoard
writer.close()
