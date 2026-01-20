#!/usr/bin/env python3
"""
Test CIFAR-10 XNNPACK Model Inference
Quick test to verify the model works correctly
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from executorch.extension.pybindings.portable_lib import _load_for_executorch
import time
import sys
import os

# CIFAR-10 classes
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def load_model(model_path='./models/cifar10_xnnpack.pte'):
    """Load ExecuTorch model with XNNPACK"""
    print(f"Loading model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found: {model_path}")
        sys.exit(1)
    
    model = _load_for_executorch(model_path)
    
    # Check file size
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"✓ Model loaded successfully")
    print(f"  Size: {size_mb:.2f} MB")
    
    return model


def predict_image(model, image_path):
    """Run inference on a single image"""
    
    # Load and preprocess image
    print(f"\nProcessing image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return None
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Warmup (5 runs)
    print("Warming up model (5 runs)...")
    for _ in range(5):
        _ = model.forward((input_tensor,))
    print("✓ Warmup complete")
    
    # Measure inference time
    print("\nRunning inference...")
    start_time = time.perf_counter()
    outputs = model.forward((input_tensor,))
    inference_time = (time.perf_counter() - start_time) * 1000
    
    # Process output
    output_tensor = outputs[0]
    if output_tensor.dim() == 2:
        logits = output_tensor[0]
    else:
        logits = output_tensor
    
    probabilities = torch.nn.functional.softmax(logits, dim=0)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()
    
    # Display results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"\n🎯 Prediction:")
    print(f"   Class: {CLASSES[predicted_class]}")
    print(f"   Confidence: {confidence*100:.2f}%")
    print(f"\n⚡ Performance:")
    print(f"   Inference time: {inference_time:.2f} ms")
    
    # Show top 3 predictions
    print(f"\n📊 Top 3 Predictions:")
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    for i in range(3):
        print(f"   {i+1}. {CLASSES[top3_idx[i]]}: {top3_prob[i]*100:.2f}%")
    
    print("="*60)
    
    return predicted_class, confidence, inference_time


def run_multiple_inferences(model, image_path, num_runs=20):
    """Run multiple inferences for more accurate timing"""
    
    print(f"\n\n{'='*60}")
    print(f"RUNNING {num_runs} INFERENCES FOR ACCURATE TIMING")
    print("="*60)
    
    # Prepare input
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        _ = model.forward((input_tensor,))
    
    # Run multiple times
    print(f"Running {num_runs} inferences...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.forward((input_tensor,))
        times.append((time.perf_counter() - start) * 1000)
    
    # Calculate statistics
    import numpy as np
    mean_time = np.mean(times)
    median_time = np.median(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    print("\n📊 Inference Statistics:")
    print(f"   Mean:   {mean_time:.2f} ms")
    print(f"   Median: {median_time:.2f} ms")
    print(f"   Min:    {min_time:.2f} ms")
    print(f"   Max:    {max_time:.2f} ms")
    print(f"   Std:    {std_time:.2f} ms")
    print(f"\n🚀 Expected FPS: {1000/mean_time:.1f}")
    
    # Performance evaluation
    print("\n💡 Performance Evaluation:")
    if mean_time < 15:
        print("   ✅ EXCELLENT - Model is running very fast!")
    elif mean_time < 30:
        print("   ✅ GOOD - Model performance is solid")
    elif mean_time < 100:
        print("   ⚠️  MODERATE - Performance is acceptable but not optimal")
    else:
        print("   ❌ SLOW - Model is not properly optimized")
        print("      Check if XNNPACK delegation worked correctly")
    
    print("="*60)
    
    return mean_time


def main():
    print("="*60)
    print("CIFAR-10 XNNPACK Model Inference Test")
    print("="*60)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for test images
        if os.path.exists('cat.jpg'):
            image_path = 'cat.jpg'
        else:
            print("\n❌ No image provided!")
            print("Usage: python test_inference.py <image_path>")
            print("Example: python test_inference.py cat.jpg")
            sys.exit(1)
    
    # Load model
    model = load_model('./models/cifar10_xnnpack.pte')
    
    # Single inference test
    predicted_class, confidence, inf_time = predict_image(model, image_path)
    
    if predicted_class is not None:
        # Multiple inference test
        mean_time = run_multiple_inferences(model, image_path, num_runs=20)
        
        print("\n\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print(f"\n✓ Model is working correctly")
        print(f"✓ Predicted: {CLASSES[predicted_class]} ({confidence*100:.2f}%)")
        print(f"✓ Average inference time: {mean_time:.2f} ms")
        print("\nNext step: Run benchmark_model.py for full comparison")
        print("="*60)


if __name__ == "__main__":
    main()
