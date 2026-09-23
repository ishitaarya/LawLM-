import torch
import torch.nn as nn


print("=" * 50)
print("LawLM - PyTorch Environment Test")
print("=" * 50)

# PyTorch version
print("PyTorch version:", torch.__version__)

# Device
device = torch.device("cpu")

print("Device:", device)

# Create random tensor
x = torch.randn(2, 5)

print("\nInput Tensor:")
print(x)

# Simple neural network
model = nn.Linear(5, 3)

output = model(x)

print("\nModel Output:")
print(output)

print("\nModel Parameters:")
print(sum(p.numel() for p in model.parameters()))

print("\n" + "=" * 50)
print("PyTorch test successful!")
print("=" * 50)