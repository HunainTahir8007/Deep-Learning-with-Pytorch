import torch
tensor=torch.tensor([1,2,3])
res=tensor @ tensor
print(res)
mul=tensor.matmul(tensor)
print(mul)