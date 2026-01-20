import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import time
import json
import urllib.request
import os


# Paths
MODEL_PATH = "models/mobilenet_v2.pth"
LABELS_PATH = "models/imagenet_labels.json"
IMAGE_PATH = "images/cat.jpg"

# Download and save ImageNet labels (only first time)
if not os.path.exists(LABELS_PATH):
    print("Downloading ImageNet labels...")
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(LABELS_URL) as url:
        labels = json.load(url)
    
    # Save labels locally
    with open(LABELS_PATH, 'w') as f:
        json.dump(labels, f)
    print(f"Labels saved to {LABELS_PATH}")
else:
    print("Loading labels from disk...")
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)

# Load or download model
if not os.path.exists(MODEL_PATH):
    print("Downloading MobileNetV2 model...")
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
else:
    print("Loading model from disk...")
    model = models.mobilenet_v2()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# Load and preprocess image
print(f"\nLoading image from {IMAGE_PATH}...")
img = Image.open(IMAGE_PATH)
img_tensor = preprocess(img)
batch = img_tensor.unsqueeze(0)

# Perform inference with timing
print("Running inference...")
start_time = time.time()

with torch.no_grad():
    output = model(batch)
    
inference_time = (time.time() - start_time) * 1000

# Get predictions
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_idx = torch.topk(probabilities, 5)

# Display results
print("\n" + "="*50)
print("CLASSIFICATION RESULTS")
print("="*50)
print(f"Inference Time: {inference_time:.2f} ms\n")
print("Top 5 Predictions:")
print("-"*50)

for i in range(5):
    idx = top5_idx[i].item()
    prob = top5_prob[i].item()
    print(f"{i+1}. {labels[idx]:20s} - {prob*100:.2f}%")

print("="*50)
