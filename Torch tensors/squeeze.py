import torch 
# squeeze function(it remove all the 1 dimention form a target tensor)
x=torch.tensor([[1,2,3,4,5,6,7,8,9]])
print(x)
z=x.squeeze()
print(z)
# unsqueeze add the 1 dimention
f=z.unsqueeze(dim=1)
print(f)
