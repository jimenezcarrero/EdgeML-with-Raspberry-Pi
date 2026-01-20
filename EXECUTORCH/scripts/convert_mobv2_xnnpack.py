import torch
from torchvision import models
from executorch.exir import to_edge_transform_and_lower
from torch.export import export
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# Paths
PYTORCH_MODEL_PATH = "models/mobilenet_v2.pth"
EXECUTORCH_MODEL_PATH = "models/mobilenet_v2_xnnpack.pte"

print("Loading PyTorch model...")
model = models.mobilenet_v2()
model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location='cpu'))
model.eval()

# Create example input
example_input = (torch.randn(1, 3, 224, 224),)

print("Exporting to ExecuTorch with XNNPACK backend...")

# Step 1: Export to EXIR
print("  1. Capturing model with torch.export...")
exported_program = export(model, example_input)

# Step 2: Use new API - transform and lower in one step
print("  2. Lowering to Edge with XNNPACK delegation...")
edge_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],
)

# Step 3: Convert to ExecuTorch program
print("  3. Converting to ExecuTorch...")
executorch_program = edge_program.to_executorch()

# Step 4: Save as .pte file
print("  4. Saving to .pte file...")
with open(EXECUTORCH_MODEL_PATH, "wb") as f:
    f.write(executorch_program.buffer)

print(f"\n? Model successfully exported to {EXECUTORCH_MODEL_PATH}")

# Display file size
import os
pytorch_size = os.path.getsize(PYTORCH_MODEL_PATH) / (1024 * 1024)
executorch_size = os.path.getsize(EXECUTORCH_MODEL_PATH) / (1024 * 1024)

print("\n" + "="*50)
print("MODEL SIZE COMPARISON")
print("="*50)
print(f"PyTorch model:           {pytorch_size:.2f} MB")
print(f"ExecuTorch+XNNPACK:      {executorch_size:.2f} MB")
print("="*50)