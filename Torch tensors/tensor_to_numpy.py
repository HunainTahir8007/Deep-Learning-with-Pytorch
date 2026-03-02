import torch 
tensor=torch.ones(5)
print(tensor)
print("Tensor after changing to the Numpy")
n=tensor.numpy()
print(n)
tensor.add_(5)
print(tensor)
print(n)