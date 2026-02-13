import torch 
#indexing in the torch is same as numpy 
x=torch.arange(1,10)
x=x.reshape(1,3,3)
print(x[:,1,1])
print(x[:,:,1])