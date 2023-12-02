import torch

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of the current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of the current CUDA device: {cuda_id}")

print(f"Name of the current CUDA device: {torch.cuda.get_device_name(cuda_id)}")


