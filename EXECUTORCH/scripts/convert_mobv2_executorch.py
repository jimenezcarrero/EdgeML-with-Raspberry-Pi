import torch
from torchvision import models
from executorch.exir import to_edge
from torch.export import export

# Paths
PYTORCH_MODEL_PATH = "models/mobilenet_v2.pth"
EXECUTORCH_MODEL_PATH = "models/mobilenet_v2.pte"

print("Loading PyTorch model...")
# Load the saved model
model = models.mobilenet_v2()
model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location='cpu'))
model.eval()

# Create example input (batch_size=1, channels=3, height=224, width=224)
example_input = (torch.randn(1, 3, 224, 224),)

print("Exporting to ExecuTorch format...")

# Step 1: Export to EXIR (ExecuTorch Intermediate Representation)
print("  1. Capturing model with torch.export...")
exported_program = export(model, example_input)

# Step 2: Convert to Edge dialect
print("  2. Converting to Edge dialect...")
edge_program = to_edge(exported_program)

# Step 3: Convert to ExecuTorch program
print("  3. Lowering to ExecuTorch...")
executorch_program = edge_program.to_executorch()

# Step 4: Save as .pte file
print("  4. Saving to .pte file...")
with open(EXECUTORCH_MODEL_PATH, "wb") as f:
    f.write(executorch_program.buffer)

print(f"\n? Model successfully exported to {EXECUTORCH_MODEL_PATH}")

# Display file sizes for comparison
import os
pytorch_size = os.path.getsize(PYTORCH_MODEL_PATH) / (1024 * 1024)
executorch_size = os.path.getsize(EXECUTORCH_MODEL_PATH) / (1024 * 1024)

print("\n" + "="*50)
print("MODEL SIZE COMPARISON")
print("="*50)
print(f"PyTorch model:    {pytorch_size:.2f} MB")
print(f"ExecuTorch model: {executorch_size:.2f} MB")
print(f"Reduction:        {((pytorch_size - executorch_size) / pytorch_size * 100):.1f}%")
print("="*50)