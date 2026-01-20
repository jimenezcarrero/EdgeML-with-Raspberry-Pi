import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import json
from executorch.extension.pybindings.portable_lib import _load_for_executorch

# Paths
EXECUTORCH_MODEL_PATH = "models/mobilenet_v2_quantized_xnnpack.pte"
LABELS_PATH = "models/imagenet_labels.json"
IMAGE_PATH = "images/cat.jpg"

# Load labels
print("Loading labels...")
with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)

# Load ExecuTorch model
print(f"Loading ExecuTorch model from {EXECUTORCH_MODEL_PATH}...")
model = _load_for_executorch(EXECUTORCH_MODEL_PATH)

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# Load and preprocess image
print(f"Loading image from {IMAGE_PATH}...")
img = Image.open(IMAGE_PATH)
img_tensor = preprocess(img)
batch = img_tensor.unsqueeze(0)

# Perform inference with timing
print("Running ExecuTorch inference (Quantized INT8)...")
start_time = time.time()

output = model.forward((batch,))

inference_time = (time.time() - start_time) * 1000

# Get predictions - convert to float first for quantized models
output_tensor = output[0]

# Convert to float if it's in quantized format
if output_tensor.dtype != torch.float32:
    output_tensor = output_tensor.float()

probabilities = torch.nn.functional.softmax(output_tensor[0], dim=0)
top5_prob, top5_idx = torch.topk(probabilities, 5)

# Display results
print("\n" + "="*50)
print("EXECUTORCH QUANTIZED INT8 RESULTS")
print("="*50)
print(f"Inference Time: {inference_time:.2f} ms")
print(f"Output dtype:   {output[0].dtype}\n")
print("Top 5 Predictions:")
print("-"*50)

for i in range(5):
    idx = top5_idx[i].item()
    prob = top5_prob[i].item()
    print(f"{i+1}. {labels[idx]:20s} - {prob*100:.2f}%")

print("="*50)