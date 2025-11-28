import torch
import time

# Check if CUDA is available
if not torch.cuda.is_available():
    print(" CUDA is not available. Using CPU only.")
else:
    print(" CUDA is available.")
    print("GPU in use:", torch.cuda.get_device_name(0))

# Create a large random tensor
size = 10000
x_cpu = torch.randn(size, size)

# ----- CPU computation -----
start_cpu = time.time()
y_cpu = torch.mm(x_cpu, x_cpu)
end_cpu = time.time()
print(f" CPU time: {end_cpu - start_cpu:.4f} seconds")
# ----- GPU computation -----
if torch.cuda.is_available():
    x_gpu = x_cpu.to('cuda')
    torch.cuda.synchronize()  # Ensure GPU ready

    start_gpu = time.time()
    y_gpu = torch.mm(x_gpu, x_gpu)
    torch.cuda.synchronize()  # Wait for all ops to finish
    end_gpu = time.time()

    print(f" GPU time: {end_gpu - start_gpu:.4f} seconds")
    print(" GPU is actively being used for computation.")
