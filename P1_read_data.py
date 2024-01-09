# PyTorch提供了torch.utils.data.Dataset类，可以通过继承这个类并实现相应的方法来创建自定义数据集。
from torch.utils.data import Dataset
# PIL库是一个功能强大的图像处理库，提供了许多图像操作和处理的功能。其中的Image模块是PIL库中最常用的模块之一，用于加载、处理和保存图像文件。
from PIL import Image
# os模块提供了与操作系统交互的功能，允许您访问文件系统、执行系统命令、管理进程等。
import os

# 名为Mydata的自定义数据集类，继承自Dataset类,这个数据集类的作用是加载图像数据和相应的标签
class Mydata(Dataset):

    # 初始化 提供全局变量
    # 数据集类的构造函数，您可以在这里初始化数据集的属性和变量
    def __init__(self,root_dir,label_dir):
        # 接受root_dir和label_dir作为参数，这些参数表示图像文件和标签文件所在的根目录和子目录。
        self.root_dir=root_dir
        self.label_dir=label_dir
        # 使用os.path.join函数将根目录和子目录合并成一个完整的路径
        self.path=os.path.join(self.root_dir,self.label_dir)
        # 使用os.listdir函数获取该路径下的所有图像文件名。
        # os.listdir用于返回指定目录中的所有文件和文件夹的名称列表。接受一个路径作为参数，并返回该路径下所有文件和文件夹的名称列表。
        self.img_path=os.listdir(self.path)

    # 根据给定的索引返回一个样本。您可以在这里加载和处理数据，并将其返回
    def __getitem__(self, idx):
        # 根据给定的索引idx获取对应的图像文件名
        img_name=self.img_path[idx]
        # 使用os.path.join函数构建完整的图像文件路径
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        # 使用PIL库的Image.open函数加载图像文件，并将其存储在img变量中
        img=Image.open(img_item_path)
        # 将label_dir作为标签返回
        label=self.label_dir
        return img,label

    def __len__(self):
        # 返回图像文件列表的长度，表示数据集中样本的数量
        return len(self.img_path)


# 根目录
root_dir="hymenoptera_data/train"
# label为图片上一级名称
ants_label_dir="ants"
bees_label_dir="bees"
# 创建了名为ants_dataset和bees_dataset的两个Mydata数据集实例，分别用于加载ants和bees类别的图像数据。
ants_dataset=Mydata(root_dir,ants_label_dir)
bees_dataset=Mydata(root_dir,bees_label_dir)

# 将这两个数据集实例合并成一个训练数据集train_dataset
train_dataset=ants_dataset+bees_dataset


