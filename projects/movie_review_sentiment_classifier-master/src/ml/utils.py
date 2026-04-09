from datetime import datetime
import torch
import os


def configure_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        print("CUDA not found. Check your drivers/PyTorch version!")
        return torch.device("cpu")


def get_save_dir(base_dir="./models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
