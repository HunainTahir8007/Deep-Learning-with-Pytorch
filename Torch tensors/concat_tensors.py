import torch 

data=[[1,2],[3,4]]
x=torch.tensor(data)
x_dat=torch.cat([x,x,x],dim=1)
print(x_dat)