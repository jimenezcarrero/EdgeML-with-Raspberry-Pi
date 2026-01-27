#!/usr/bin/env python3
"""
MobileNetV2 Inference Comparison: CPU vs MemryX MX3 Accelerator

This script demonstrates:
1. Loading and preprocessing an image
2. Running inference on both CPU (TensorFlow/Keras) and MX3 accelerator
3. Comparing latency and accuracy between the two approaches
4. Decoding and displaying top-5 predictions

Author: Marcelo Rovai
Date: January 2026
"""

import os
import time
import json
from pathlib import Path

import numpy as np
from PIL import Image
import requests

# Suppress TensorFlow info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from memryx import SyncAccl

# ============================================================================
# Configuration
# ============================================================================

IMAGES_DIR = Path("./images")
MODELS_DIR = Path("./models")
DFP_PATH = MODELS_DIR / "mobilenet_v2.dfp"
KERAS_PATH = MODELS_DIR / "mobilenet_v2.h5"
IMAGE_PATH = IMAGES_DIR / "cat.jpg"

IMAGENET_JSON = MODELS_DIR / "imagenet_class_index.json"
IMAGENET_JSON_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "imagenet_class_index.json"
)

# ============================================================================
# Label Management
# ============================================================================

def ensure_imagenet_labels():
    """
    Download ImageNet class labels if not already present.
    Creates models directory if needed.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if IMAGENET_JSON.exists():
        print(f"✓ ImageNet labels already exist: {IMAGENET_JSON}")
        return
    
    print("Downloading ImageNet class index...")
    try:
        resp = requests.get(IMAGENET_JSON_URL, timeout=30)
        resp.raise_for_status()
        IMAGENET_JSON.write_bytes(resp.content)
        print(f"✓ Saved ImageNet labels to: {IMAGENET_JSON}")
    except Exception as e:
        print(f"✗ Error downloading labels: {e}")
        raise


def load_idx2label():
    """
    Load ImageNet class labels from JSON file.
    
    Returns:
        list: List of 1000 class labels indexed by class ID
    """
    with open(IMAGENET_JSON, "r") as f:
        class_idx = json.load(f)
    
    # Convert dict to list indexed by class ID
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    return idx2label


# ============================================================================
# Image Preprocessing
# ============================================================================

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for MobileNetV2 inference.
    
    MobileNetV2 expects:
    - Input shape: (224, 224, 3)
    - Batch dimension: (1, 224, 224, 3)
    - Preprocessing: Pixels scaled to [-1, 1]
    - Color format: RGB
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image with shape (1, 224, 224, 3)
    """
    # Load image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    
    # Resize to model's expected input size
    img = img.resize((224, 224))
    
    # Convert to numpy array
    arr = np.array(img).astype(np.float32)
    
    # Apply MobileNetV2-specific preprocessing
    # This normalizes pixels to the range [-1, 1]
    arr = keras.applications.mobilenet_v2.preprocess_input(arr)
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    arr = np.expand_dims(arr, 0)
    
    return arr


# ============================================================================
# Inference Result Processing
# ============================================================================

def topk_from_probs(probs, k=5):
    """
    Extract top-k predictions from probability/logit array.
    
    Args:
        probs: Probability or logit array, shape (1, num_classes) or (num_classes,)
        k: Number of top predictions to return
        
    Returns:
        list: List of (class_index, probability) tuples, sorted by probability
    """
    probs = np.array(probs)
    
    # Remove batch dimension if present
    if probs.ndim == 2:
        probs = probs[0]
    
    # If outputs are logits (not probabilities), apply softmax
    # Uncomment the following line if needed:
    # probs = tf.nn.softmax(probs).numpy()
    
    # Normalize to sum to 1 (convert to probabilities if needed)
    s = probs.sum()
    if s > 0:
        probs = probs / s
    
    # Get indices of top-k classes
    idxs = np.argsort(probs)[::-1][:k]
    
    return [(int(i), float(probs[i])) for i in idxs]


def label_for(idx, idx2label):
    """
    Get human-readable label for a class index.
    
    Args:
        idx: Class index
        idx2label: List of class labels
        
    Returns:
        str: Human-readable class label
    """
    if idx2label is not None and idx < len(idx2label):
        return idx2label[idx]
    return f"class_{idx}"


def print_top5(top5, idx2label, prefix="Top-5"):
    """
    Print top-5 predictions in a formatted way.
    
    Args:
        top5: List of (class_index, probability) tuples
        idx2label: List of class labels
        prefix: Prefix for the output
    """
    print(f"\n{prefix}:")
    for idx, prob in top5:
        name = label_for(idx, idx2label)
        print(f"  #{idx:4d}: {name:20s}  ({prob*100:.1f}%)")


# ============================================================================
# Main Inference Pipeline
# ============================================================================

def main():
    """
    Main inference pipeline comparing CPU and MX3 accelerator performance.
    """
    print("=" * 70)
    print("MobileNetV2 Inference: CPU vs MemryX MX3 Comparison")
    print("=" * 70)
    
    # Ensure labels are available
    ensure_imagenet_labels()
    idx2label_full = load_idx2label()
    print(f"✓ Loaded {len(idx2label_full)} ImageNet class labels")
    
    # Load and preprocess image
    print(f"\n✓ Loading image: {IMAGE_PATH}")
    x = load_and_preprocess_image(IMAGE_PATH)
    print(f"✓ Preprocessed image shape: {x.shape}")
    
    # ========================================================================
    # CPU Inference
    # ========================================================================
    print("\n" + "-" * 70)
    print("CPU INFERENCE (TensorFlow/Keras)")
    print("-" * 70)
    
    print(f"Loading model: {KERAS_PATH}")
    cpu_model = keras.models.load_model(KERAS_PATH)
    
    # Warm-up run (first inference includes model loading overhead)
    print("Running warm-up inference...")
    _ = cpu_model.predict(x, verbose=0)
    
    # Timed inference
    print("Running timed inference...")
    start = time.time()
    cpu_outputs = cpu_model.predict(x, verbose=0)
    cpu_latency = time.time() - start
    
    print(f"\n✓ CPU Latency: {cpu_latency*1000:.2f} ms")
    
    # Get predictions
    num_classes = cpu_outputs.shape[-1]
    idx2label = idx2label_full if num_classes == len(idx2label_full) else None
    
    cpu_top5 = topk_from_probs(cpu_outputs, k=5)
    print_top5(cpu_top5, idx2label, prefix="CPU Predictions")
    
    # ========================================================================
    # MX3 Accelerator Inference
    # ========================================================================
    print("\n" + "-" * 70)
    print("MX3 ACCELERATOR INFERENCE (MemryX)")
    print("-" * 70)
    
    print(f"Loading DFP: {DFP_PATH}")
    accl = SyncAccl(dfp=str(DFP_PATH))
    
    # Warm-up run
    print("Running warm-up inference...")
    _ = accl.run(x)
    
    # Timed inference
    print("Running timed inference...")
    start = time.time()
    mxa_outputs = accl.run(x)
    mxa_latency = time.time() - start
    
    print(f"\n✓ MXA Latency: {mxa_latency*1000:.2f} ms")
    
    # Convert to numpy array and normalize shape
    mxa_outputs = np.array(mxa_outputs)
    if mxa_outputs.ndim == 3:
        mxa_outputs = mxa_outputs[0]
    
    # Get predictions
    mxa_top5 = topk_from_probs(mxa_outputs, k=5)
    print_top5(mxa_top5, idx2label, prefix="MXA Predictions")
    
    # Clean up
    accl.shutdown()
    
    # ========================================================================
    # Performance Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    speedup = cpu_latency / mxa_latency
    print(f"\nCPU Latency:        {cpu_latency*1000:>8.2f} ms")
    print(f"MXA Latency:        {mxa_latency*1000:>8.2f} ms")
    print(f"Speedup (CPU/MXA):  {speedup:>8.2f}x")
    
    # Compare top predictions
    cpu_top1 = cpu_top5[0][0]
    mxa_top1 = mxa_top5[0][0]
    
    if cpu_top1 == mxa_top1:
        print(f"\n✓ Both models agree on top prediction: {label_for(cpu_top1, idx2label)}")
    else:
        print(f"\n⚠ Different top predictions:")
        print(f"  CPU: {label_for(cpu_top1, idx2label)}")
        print(f"  MXA: {label_for(mxa_top1, idx2label)}")
    
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
