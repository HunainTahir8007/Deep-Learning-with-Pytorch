import torch

device='cuda' if torch.cuda.is_available() else 'cpu'
tensor=torch.tensor([1,2,3],device='cpu')
print(tensor)
print(tensor.device)
# now shifting cpu to cuda 
tensor_gpu=tensor.to(device)
print(tensor_gpu.device)

tensor_cpu=tensor_gpu.cpu().numpy()
print(tensor_cpu.device)