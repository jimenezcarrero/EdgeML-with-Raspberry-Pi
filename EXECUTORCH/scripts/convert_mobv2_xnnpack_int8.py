import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

PYTORCH_MODEL_PATH = "models/mobilenet_v2.pth"
EXECUTORCH_QUANTIZED_PATH = "models/mobilenet_v2_quantized_xnnpack.pte"
CALIB_IMAGES_DIR = "calib_images"   # <-- put some natural images here

# 1) Load FP32 model
model = models.mobilenet_v2()
model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location="cpu"))
model.eval()

# Example input only defines shapes for export
example_inputs = (torch.randn(1, 3, 224, 224),)

# 2) Configure XNNPACK quantizer (global symmetric config)
qparams = get_symmetric_quantization_config(is_per_channel=True)
quantizer = XNNPACKQuantizer()
quantizer.set_global(qparams)

# 3) Export float model for PT2E and prepare for quantization
exported = torch.export.export(model, example_inputs)
training_ep = exported.module()
prepared = prepare_pt2e(training_ep, quantizer)

# 4) Calibration with REAL images using SAME preprocessing as inference
calib_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

calib_dataset = datasets.ImageFolder(CALIB_IMAGES_DIR, transform=calib_transform)
calib_loader = torch.utils.data.DataLoader(
    calib_dataset, batch_size=1, shuffle=True
)

print(f"Calibrating on {len(calib_dataset)} images from {CALIB_IMAGES_DIR}...")

num_calib = min(100, len(calib_dataset))  # or adjust
with torch.no_grad():
    for i, (calib_img, _) in enumerate(calib_loader):
        if i >= num_calib:
            break
        prepared(calib_img)

# 5) Convert calibrated model to quantized model
quantized_model = convert_pt2e(prepared)

# 6) Export quantized model and lower to XNNPACK, then to ExecuTorch
exported_quant = export(quantized_model, example_inputs)

et_program = to_edge_transform_and_lower(
    exported_quant,
    partitioner=[XnnpackPartitioner()],
).to_executorch()

# 7) Save .pte and compute sizes
with open(EXECUTORCH_QUANTIZED_PATH, "wb") as f:
    et_program.write_to_file(f)

pytorch_size = os.path.getsize(PYTORCH_MODEL_PATH) / (1024 * 1024)
quantized_size = os.path.getsize(EXECUTORCH_QUANTIZED_PATH) / (1024 * 1024)

print("\n" + "="*60)
print("MODEL SIZE COMPARISON")
print("="*60)
print(f"PyTorch (FP32):                  {pytorch_size:6.2f} MB")
print(f"ExecuTorch Quantized (INT8):     {quantized_size:6.2f} MB")
print(f"Size reduction:                  {((pytorch_size - quantized_size) / pytorch_size * 100):5.1f}%")
print(f"Savings:                         {pytorch_size - quantized_size:6.2f} MB")
print("="*60)
