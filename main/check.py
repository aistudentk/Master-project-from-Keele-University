# import os

# dataset_dir = "D:\Keele\CSC40040毕业设计\Project\evalu\DIC_crack_dataset\Test"
# if os.path.isdir(dataset_dir):
#     print(f"Directory exists: {dataset_dir}")
# else:
#     print(f"Directory does not exist: {dataset_dir}")

import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available")

# 你的其他代码
# ...