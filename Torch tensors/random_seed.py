import torch 

tensor_A=torch.rand(1,50)
tensor_B=torch.rand(1,50)
# print(tensor_A)
# print(tensor_B)
print(tensor_A==tensor_B)# this will bot crete the same values every time 

#using the random seed
random_seed=42
torch.manual_seed(random_seed)
tensor_C=torch.rand(1,50)
torch.manual_seed(random_seed)
tensor_D=torch.rand(1,50)
print(tensor_C==tensor_D)