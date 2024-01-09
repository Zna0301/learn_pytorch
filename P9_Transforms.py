from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 通过transforms.ToTensor解决两个问题
# 1.transforms使用
# 2.为什么用ToTensor数据类型

# 1.transforms使用
# 读取图片
img_path="hymenoptera_data/train/ants/0013035.jpg"
# 使用PIL.Image.open打开图像文件，并将其赋值给img变量
img=Image.open(img_path)
# print(img)

# 导入ToTensor变换方法，并创建了一个transform对象tensor_trans
tensor_trans=transforms.ToTensor()
# 使用transform对象tensor_trans将图像转换为张量，结果保存在tensor_img变量中。
tensor_img=tensor_trans(img)
# print(tensor_img)


# 2.为什么用ToTensor数据类型
# 通过将图像转换为张量，可以方便地将其输入到神经网络中进行训练和推理。
writer=SummaryWriter("log1s")
writer.add_image("Tensor_img",tensor_img)
writer.close()


