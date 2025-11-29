import torch

tensor = torch.ones((2, 2))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor = tensor.to(device)

print(tensor)
print("Device:", tensor.device)