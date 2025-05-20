# VENV name: cudaconfig
import torch

print("CUDA is available: ", torch.cuda.is_available())
# True

print("Number of CUDA devices: ", torch.cuda.device_count())
# 5

print("CUDA current device: ", torch.cuda.current_device())
# 0

print("CUDA device name: ", torch.cuda.get_device_name(0))
# 'NVIDIA GeForce RTX 3060'