import torch
x=torch.randn(225,225,3)
print(x.shape)

#niw we change the shape by permulte
z=x.permute(2,1,0)
print(z.shape)