import torch
import torch.nn.functional as F
# 二维卷积操作
# 这个例子中的卷积操作是在一个输入图像上应用一个 3x3 的卷积核，步幅为 1。
# 卷积核从左上角开始滑动，对应位置的输出值是卷积核与输入图像局部区域的逐元素乘积的和。

# 1.输入图像和卷积核的定义
# 输入图像 5维矩阵 5*5
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0,],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])
# 卷积核3*3
kernal=torch.tensor([[1,2,1],
                     [0,1,0],
                     [2,1,0]])
print(input.shape)# torch.Size([5, 5])
print(kernal.shape)# torch.Size([3, 3])


# 2.尺寸转换 将输入图像和卷积核的维度进行转换，以符合 PyTorch 中 F.conv2d 函数的输入要求
# 即将它们的维度变为 (batch_size, channels, height, width)。
input=torch.reshape(input,(1,1,5,5))
kernal=torch.reshape(kernal,(1,1,3,3))
print(input.shape)# torch.Size([1, 1, 5, 5])
print(kernal.shape)# torch.Size([1, 1, 3, 3])


# 3.卷积操作：使用 F.conv2d 进行卷积操作，其中 stride=1 表示卷积核的步幅为 1。
output=F.conv2d(input,kernal,stride=1)
# 卷积操作的输出结果，其中每个值表示卷积核在输入图像上滑动时的计算结果
print(output)# tensor([[[[10, 12, 12],
                        #[18, 16, 16],
                        #[13,  9,  3]]]])

# 4.stride=2 表示卷积核的步幅为 2
# 这导致输出特征图的空间尺寸减小。步幅的选择可以根据任务和网络结构进行调整，通常较大的步幅会减小输出特征图的尺寸。
output2=F.conv2d(input,kernal,stride=2)
print(output2)# tensor([[[[10, 12],
                        #[13,  3]]]])


# 5.padding（填充）
output3=F.conv2d(input,kernal,stride=1,padding=1)
print(output3)
# tensor([[[[ 1,  3,  4, 10,  8],
#           [ 5, 10, 12, 12,  6],
#           [ 7, 18, 16, 16,  8],
#           [11, 13,  9,  3,  4],
#           [14, 13,  9,  7,  4]]]])