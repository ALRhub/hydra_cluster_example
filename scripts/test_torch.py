import torch
print("Torch version:", torch.__version__)
if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")