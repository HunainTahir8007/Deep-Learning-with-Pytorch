import torch
tensor=torch.arange(0,100,10)
# arg min max return the index where the max and min value 
print(tensor.argmax())
print(tensor.argmin())