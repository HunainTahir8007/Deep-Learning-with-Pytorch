import torch 

tensor=torch.arange(0,100,10)

print(tensor.max())
print(tensor.min())
print(tensor.sum())

#for the means the type of the tensor is float 32 so we have to change the datatype
print(tensor.type()) # this long in tensoe we have to change the type
print(torch.mean(tensor.type(torch.float32)))

