import numpy as np
from PIL import Image, ImageDraw, ImageFont
from memryx import AsyncAccl
import time
from collections import deque

# COCO dataset class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def preprocess_image(image_path, input_size=640):
    """
    Preprocess image for YOLOv8 inference.
    
    Returns: input_tensor, original_image, ratio, (pad_w, pad_h)
    """
    # Load image
    original = Image.open(image_path).convert('RGB')
    w, h = original.size
    
    # Calculate scaling ratio
    ratio = min(input_size / h, input_size / w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    
    # Resize
    resized = original.resize((new_w, new_h), Image.BILINEAR)
    
    # Create padded image (gray background)
    padded = Image.new('RGB', (input_size, input_size), (114, 114, 114))
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2
    padded.paste(resized, (pad_w, pad_h))
    
    # Convert to tensor [1, 3, 640, 640]
    img_array = np.array(padded).astype(np.float32) / 255.0
    img_tensor = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_tensor = np.expand_dims(img_tensor, axis=0)   # Add batch dimension
    
    return img_tensor, original, ratio, (pad_w, pad_h)

def compute_iou_batch(box, boxes):
    """Compute IoU between one box and multiple boxes."""
    # Intersection
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Union
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    
    return intersection / np.maximum(union, 1e-6)

def apply_nms(boxes, scores, iou_threshold=0.45):
    """Apply Non-Maximum Suppression."""
    indices = np.argsort(scores)[::-1]  # Sort by score descending
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = compute_iou_batch(boxes[current], boxes[indices[1:]])
        
        # Keep only boxes with IoU below threshold
        indices = indices[1:][ious < iou_threshold]
    
    return keep

def decode_predictions(output, conf_threshold=0.25, iou_threshold=0.45):
    """
    Decode YOLOv8 predictions.
    
    Input format: (1, 84, 8400) where 84 = 4 bbox + 80 classes
    Returns: detections array [x1, y1, x2, y2, confidence, class_id]
    """
    # Transpose from (1, 84, 8400) to (8400, 84)
    predictions = np.transpose(output, (0, 2, 1))[0]
    
    # Split boxes and scores
    boxes_xywh = predictions[:, :4]    # [x_center, y_center, width, height]
    class_scores = predictions[:, 4:]  # 80 class scores
    
    # Get best class for each prediction
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    # Filter by confidence threshold
    mask = max_scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes_xywh) == 0:
        return np.array([])
    
    # Convert from xywh to xyxy format
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
    
    # Apply NMS
    keep_indices = apply_nms(boxes_xyxy, max_scores, iou_threshold)
    
    # Build final detections
    detections = np.column_stack([
        boxes_xyxy[keep_indices],
        max_scores[keep_indices, np.newaxis],
        class_ids[keep_indices, np.newaxis]
    ])
    
    return detections

def scale_boxes_to_original(boxes, ratio, pad, original_size):
    """Scale boxes from model coordinates to original image coordinates."""
    if len(boxes) == 0:
        return boxes
    
    pad_w, pad_h = pad
    orig_w, orig_h = original_size
    
    # Remove padding
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    
    # Scale to original size
    boxes[:, :4] /= ratio
    
    # Clip to image boundaries
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    
    return boxes

def draw_detections(image, detections):
    """Draw bounding boxes and labels on image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)
        
        if class_id < 0 or class_id >= len(COCO_CLASSES):
            continue
        
        # Generate consistent color for this class
        np.random.seed(class_id)
        color = tuple(np.random.randint(50, 255, 3).tolist())
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"{COCO_CLASSES[class_id]}: {conf:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        label_y = max(y1 - text_h - 10, 0)
        draw.rectangle([x1, label_y, x1 + text_w + 10, label_y + text_h + 10], fill=color)
        draw.text((x1 + 5, label_y + 5), label, fill=(255, 255, 255), font=font)
    
    return annotated

def detect_objects(dfp_path, post_model_path, image_path, conf_threshold=0.25):
    """
    Main detection function.
    
    Args:
        dfp_path: Path to compiled .dfp model
        post_model_path: Path to post-processing .onnx model
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
    
    Returns:
        detections: Array of detections
        annotated_image: Image with drawn detections
        inference_time: Time in milliseconds
    """
    # Preprocess image
    print(f"Preprocessing: {image_path}")
    input_tensor, original_image, ratio, pad = preprocess_image(image_path)
    
    # Initialize accelerator
    print(f"Loading model: {dfp_path}")
    accl = AsyncAccl(dfp_path)
    accl.set_postprocessing_model(post_model_path, model_idx=0)
    
    # Storage for results
    results = deque()
    frame_queue = deque([input_tensor])
    
    # Define input generator
    def generate_frame():
        while len(frame_queue) > 0:
            yield frame_queue.popleft()
    
    # Define output processor
    def process_output(*outputs):
        results.append(outputs)
    
    # Run inference
    accl.connect_input(generate_frame)
    accl.connect_output(process_output)
    
    print("Running inference...")
    start_time = time.time()
    accl.wait()
    inference_time = (time.time() - start_time) * 1000
    
    print(f"Inference time: {inference_time:.2f} ms")
    
    # Get results
    if len(results) == 0:
        print("No results!")
        return np.array([]), original_image, inference_time
    
    outputs = results.popleft()
    output = outputs[0]  # Get the first (and only) output
    
    # Decode predictions
    print(f"Output shape: {output.shape}")
    detections = decode_predictions(output, conf_threshold=conf_threshold)
    
    # Scale to original image size
    if len(detections) > 0:
        detections = scale_boxes_to_original(detections, ratio, pad, original_image.size)
    
    print(f"Found {len(detections)} detections")
    
    # Draw detections
    annotated_image = draw_detections(original_image, detections)
    
    return detections, annotated_image, inference_time

# Main execution
if __name__ == "__main__":
    # Configuration
    DFP_PATH = "./models/yolov8n.dfp"
    POST_MODEL_PATH = "./models/yolov8n_post.onnx"
    IMAGE_PATH = "./images/bus.jpg"
    CONF_THRESHOLD = 0.25
    
    # Run detection
    detections, annotated_image, inference_time = detect_objects(
        DFP_PATH,
        POST_MODEL_PATH,
        IMAGE_PATH,
        CONF_THRESHOLD
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Detection Results:")
    print(f"{'='*60}")
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, class_id = det
        print(f"  {i+1}. {COCO_CLASSES[int(class_id)]}: {conf:.3f}")
        print(f"      Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    
    # Save annotated image
    if len(detections) > 0:
        output_path = IMAGE_PATH.rsplit('.', 1)[0] + '_detected.jpg'
        annotated_image.save(output_path)
        print(f"\nSaved: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Total: {len(detections)} objects")
    print(f"Time: {inference_time:.2f} ms")
    print(f"{'='*60}")
