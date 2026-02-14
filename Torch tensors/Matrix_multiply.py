import torch 
import numpy as np 

tensor=torch.ones(4,4)
print(tensor)
mut=tensor @ tensor.T
print(mut)

mul=torch.matmul(tensor,tensor.T)
print(mul)
print('-------------------Z1,Z2--------------------')
z1=tensor*tensor
print(z1)
z2=tensor.mul(tensor)
print(z2)
print('----------------Z3-----------------------')
z3=torch.rand_like(tensor)
print(z3)
torch.mul(tensor,tensor,out=z3)
print(z3)