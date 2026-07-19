import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"Is CUDA (NVIDIA GPU) available? : {torch.cuda.is_available()}")
print(f"Is MPS (Apple Mac GPU) available?: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

if torch.cuda.is_available():
    print(f"Active GPU Device Name: {torch.cuda.get_device_name(0)}")
