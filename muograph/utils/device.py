import torch

# Check if GPU is available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU is available. Using {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("GPU is not available. Using CPU")
