import subprocess
import os

def text_to_speech_piper(text, output_file="piper_output.wav"):
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

# converting text to sound:
txt = "Lilongwe is the capital of Malawi. Would you like to know more about Lilongwe or Malawi in general?"

if text_to_speech_piper(txt):
    print("You can now play the file with: aplay piper_output.wav")
else:
    print("Failed to generate speech")       