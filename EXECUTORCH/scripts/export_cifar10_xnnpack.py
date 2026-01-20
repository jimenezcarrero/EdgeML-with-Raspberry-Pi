#!/usr/bin/env python3
"""
Export CIFAR-10 Model with XNNPACK Delegation for ExecuTorch
Optimized for Raspberry Pi 5

This is the CORRECT way to export for ARM devices!
Uses XNNPACK backend for hardware acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
import os

# ============================================================================
# Model Definition (same as training)
# ============================================================================

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Block 3
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Fully connected
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten and FC
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# Export Function with XNNPACK
# ============================================================================

def export_cifar10_xnnpack(model_path='./models/cifar10_model_jit.pt'):
    """
    Export CIFAR-10 model with XNNPACK delegation
    
    Args:
        model_path: Path to trained TorchScript model
    
    Returns:
        Path to exported .pte file
    """
    
    print("="*70)
    print("CIFAR-10 Export with XNNPACK Delegation")
    print("="*70)
    
    # Check if input model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found: {model_path}")
        return None
    
    print(f"\n📂 Loading model: {model_path}")
    
    # Load the trained model
    try:
        model_jit = torch.jit.load(model_path, map_location='cpu')
        model_jit.eval()
        
        # Extract state dict and load into fresh model
        state_dict = model_jit.state_dict()
        model = CIFAR10Net()
        model.load_state_dict(state_dict)
        model.eval()
        
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Example input (CIFAR-10 images are 32x32)
    example_input = (torch.randn(1, 3, 32, 32),)
    
    print("\n🔧 Exporting to ExecuTorch with XNNPACK...")
    
    try:
        # Step 1: Export to ATEN dialect
        print("  Step 1/4: Exporting to ATEN dialect...")
        aten_dialect = export(model, example_input)
        print("  ✓ ATEN export complete")
        
        # Step 2: Convert to Edge dialect
        print("  Step 2/4: Converting to Edge dialect...")
        edge_program = to_edge(aten_dialect)
        print("  ✓ Edge conversion complete")
        
        # Step 3: Partition for XNNPACK backend (THIS IS THE KEY!)
        print("  Step 3/4: Partitioning for XNNPACK backend...")
        edge_program = edge_program.to_backend(XnnpackPartitioner())
        print("  ✓ XNNPACK partitioning complete")
        
        # Step 4: Convert to ExecuTorch program
        print("  Step 4/4: Converting to ExecuTorch program...")
        executorch_program = edge_program.to_executorch()
        print("  ✓ ExecuTorch conversion complete")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Save the model
    output_path = "./models/cifar10_xnnpack.pte"
    
    try:
        with open(output_path, "wb") as f:
            f.write(executorch_program.buffer)
        
        size_mb = len(executorch_program.buffer) / 1024 / 1024
        
        print("\n" + "="*70)
        print("✅ EXPORT SUCCESSFUL!")
        print("="*70)
        print(f"\n📦 Model saved: {output_path}")
        print(f"📊 Size: {size_mb:.2f} MB")
        
        print("\n⚡ Expected Performance on Raspberry Pi 5:")
        print("  Inference time: ~8-15 ms")
        print("  Throughput: ~65-125 FPS")
        print("  Speed-up vs PyTorch: 1.5-2x faster")
        
        print("\n🚀 Next Steps:")
        print("  1. Test the model:")
        print("     python test_inference.py")
        print("  2. Run benchmark:")
        print("     python benchmark_model.py")
        
        print("="*70)
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ Failed to save model: {e}")
        return None


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Get model path from argument or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "./models/cifar10_model_jit.pt"
    
    # Export with XNNPACK
    output_path = export_cifar10_xnnpack(model_path)
    
    if output_path:
        print(f"\n✓ Done! Use {output_path} for inference on Raspberry Pi")
    else:
        print("\n✗ Export failed. Check error messages above.")
        sys.exit(1)
