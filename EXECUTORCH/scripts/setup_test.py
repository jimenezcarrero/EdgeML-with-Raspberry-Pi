import torch
import numpy as np
from PIL import Image
import executorch

print("=" * 50)
print("SETUP VERIFICATION")
print("=" * 50)

# Check versions
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"PIL version: {Image.__version__}")
print(f"EXECUTORCH available: {executorch is not None}")

# Test basic PyTorch functionality
x = torch.randn(3, 224, 224)
print(f"\nCreated test tensor with shape: {x.shape}")

# Test PIL
test_img = Image.new('RGB', (224, 224), color='red')
print(f"Created test PIL image: {test_img.size}")

print("\n✓ Setup verification complete!")
print("=" * 50)