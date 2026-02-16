import torch

tensor_A=torch.tensor([[1,2],
                       [3,4],
                       [5,6]])
tensor_B=torch.tensor([[7,8],
                       [9,10],
                       [11,12]])

print(torch.matmul(tensor_A,tensor_B.T))
