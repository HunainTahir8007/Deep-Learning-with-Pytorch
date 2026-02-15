import torch 
import numpy as np 

data =[[1,2],[3,4]]
x_data=torch.tensor(data)
x=torch.ones_like(x_data)

y=torch.rand_like(x_data,dtype=torch.float)
print("like tensors",x)
print("random tensors",y)
