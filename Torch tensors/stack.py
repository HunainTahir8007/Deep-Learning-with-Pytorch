import torch 
x=torch.arange(1,10)

print(torch.stack((x,x,x,x),dim=0))
print(torch.stack((x,x,x,x),dim=1))