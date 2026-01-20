import numpy as np
from picamera2 import Picamera2
import time

print(f"NumPy version: {np.__version__}")

# Initialize camera
picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size":(640,480)}) 
picam2.configure(config)
picam2.start()

# Wait for camera to warm up
time.sleep(2)

print("Camera working in isolated venv!")

# Capture image
picam2.capture_file("camera_capture.jpg")
print("Image captured: camera_capture.jpg")

# Stop camera
picam2.stop()
picam2.close()
