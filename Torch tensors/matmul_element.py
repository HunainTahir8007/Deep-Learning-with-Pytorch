import torch 

tensor=torch.tensor([1,2,3])

print("Multiplt",tensor*tensor)
mul=tensor.__mul__(tensor)
print(mul)