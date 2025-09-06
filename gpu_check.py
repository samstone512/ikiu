# gpu_check.py
# A simple script to verify the PyTorch CUDA installation.

import torch

print("--- GPU Verification Script ---")
print(f"PyTorch Version: {torch.__version__}")

# Check if CUDA (GPU support) is available
is_available = torch.cuda.is_available()
print(f"Is CUDA available? -> {is_available}")

if is_available:
    # Get the number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")

    # Get the name of the current GPU
    current_gpu_name = torch.cuda.get_device_name(0)
    print(f"Current GPU Name: {current_gpu_name}")
else:
    print("\n[WARNING]: PyTorch cannot detect your GPU.")
    print("Please ensure you have installed the correct PyTorch version for your CUDA driver.")
    print("Visit: https://pytorch.org/get-started/locally/")

print("--- Verification Complete ---")
# test with this command
# python gpu_check.py