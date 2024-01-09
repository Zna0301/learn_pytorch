import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs=torch.tensor([1,2,3],dtype=torch.float)
targets=torch.tensor([1,2,5])

inputs=torch.reshape(inputs,(1,1,1,3))
targets=torch.reshape(targets,(1,1,1,3))

# loss=L1Loss()
loss=L1Loss(reduction='sum')
result=loss(inputs,targets)
print(result)# tensor(0.6667)->tensor(2.)

# MSELOSS是指均方误差损失（Mean Squared Error Loss）
lossMSE=MSELoss()
result_mse=lossMSE(inputs,targets)
print(result_mse)# tensor(1.3333)


# CrossEntropyLoss
x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])
x=torch.reshape(x,(1,3))
loss_cross=CrossEntropyLoss()
result_cross=loss_cross(x,y)
print(result_cross)# tensor(1.1019)