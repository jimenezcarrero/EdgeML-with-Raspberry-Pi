#!/usr/bin/env python3
"""
Image-to-Speech AI Pipeline for Raspberry Pi 5

Complete pipeline: Camera Image → Caption with SVM → TTS → Audio Output

run it with: python img_caption_speech.py 2>/dev/null
"""


import os
import time
import subprocess
import ollama
from picamera2 import Picamera2


def capture_image(img_path):
	# Initialize camera
	picam2 = Picamera2()
	picam2.start()

	# Wait for camera to warm up
	time.sleep(2)

	# Capture image
	picam2.capture_file(img_path)
	print("\n==> Image captured: "+img_path)

	# Stop camera
	picam2.stop()
	picam2.close()


def image_description(img_path, model):

    print ("\n==> WAIT, SVL Model working ...")
    with open(img_path, 'rb') as file:
        response = ollama.chat(
            model=model,
            messages=[
              {
                'role': 'user',
                'content': '''return the description of the image''',
                'images': [file.read()],
              },
            ],
            options = {
              'temperature': 0,
              }
      )
    return response['message']['content']


def text_to_speech_piper(text, output_file="assistant_response.wav"):
    """
    Convert text to speech using PIPER and save to WAV file
    
    Args:
        text (str): Text to convert to speech
        output_file (str): Output WAV file path
    """
    # Path to your voice model
    model_path = "voices/en_US-lessac-low.onnx"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        # Run PIPER command
        process = subprocess.Popen(
            ['piper', '--model', model_path, '--output_file', output_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send text to PIPER
        stdout, stderr = process.communicate(input=text)
        
        if process.returncode == 0:
            print(f"\nSpeech generated successfully: {output_file}")
            return True
        else:
            print(f"Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"Error running PIPER: {e}")
        return False

def play_audio(filename="assistant_response.wav"):
    """
    Play the generated speech audio
    
    Uses aplay (ALSA player) which is standard on Raspberry Pi.
    This completes the pipeline by actually speaking the response.
    
    Args:
        filename (str): Audio file to play
    
    Returns:
        bool: True if playback successful, False otherwise
    """
    try:
        # Use aplay to play the audio file
        result = subprocess.run(['aplay', filename], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            print("\nAudio playback completed")
            return True
        else:
            print(f"\nPlayback error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"\nError playing audio: {e}")
        return False


# Example usage and testing functions
if __name__ == "__main__":

	print("\n============= Image to Speech AI Pipeline Ready ===========")
	print("Press [Enter] to capture an image and voice caption it ...")

	# Step 1: Wait for user to initiate recording
	input("Press Enter to start ...")

	IMG_PATH = "/home/mjrovai/Documents/OLLAMA/SST/capt_image.jpg"
	MODEL = "moondream:latest"

	capture_image(IMG_PATH)
	caption = image_description(IMG_PATH, MODEL)
	print ("\n==> AI Response:", caption)

	text_to_speech_piper(caption)
	play_audio()


    