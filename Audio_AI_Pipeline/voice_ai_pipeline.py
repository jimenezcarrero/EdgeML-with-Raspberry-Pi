#!/usr/bin/env python3
"""
Speech-to-SLM Pipeline for Raspberry Pi 5 with TTS
Uses pyaudio for audio capture, moonshine_onnx for STT, Ollama for LLM processing,
and Piper for TTS output

Complete pipeline: User Speech → STT → OLLAMA → TTS → Audio Output

run it with: python voice_ai_pipeline.py 2>/dev/null
"""

import subprocess
import os
import pyaudio
import wave
import moonshine_onnx
import ollama
import time

# Audio configuration constants
FORMAT = pyaudio.paInt16
CHANNELS = 1    
RATE = 16000    # 16 kHz sample rate - this matches what Moonshine expects
CHUNK = 1024
DEVICE_INDEX = 2   # Your USB mic index from verify_usb_index.py

def detect_audio_devices():
    """
    Helper function to show available audio devices
    Useful for troubleshooting if your USB mic index changes
    """
    print("Available audio devices:")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"{i}: {info.get('name')} - {info.get('maxInputChannels')} input channels")
    p.terminate()

def record_audio_with_silence_detection(silence_threshold=0.01, silence_duration=2.0, 
                                       max_duration=30, output_filename="user_input.wav"):
    """
    Capture audio from USB microphone with intelligent silence detection
    
    This advanced recording function mimics how humans naturally detect the end
    of speech. Instead of recording for a fixed time, it listens continuously
    and stops when it detects a sustained period of silence, indicating the
    speaker has finished their thought.
    
    The technical approach:
    1. Continuously capture audio in small chunks (like taking snapshots)
    2. Calculate the volume (amplitude) of each chunk
    3. If volume stays below threshold for silence_duration, stop recording
    4. Include safety maximum to prevent infinite recording
    
    Args:
        silence_threshold (float): Volume level below which we consider it "silence"
                                 (0.01 is quite sensitive, 0.1 would be less sensitive)
        silence_duration (float): How many seconds of silence before stopping
        max_duration (int): Maximum recording time as safety fallback
        output_filename (str): WAV file to save the recording
    
    Returns:
        bool: True if recording successful, False otherwise
    """
    try:
        audio = pyaudio.PyAudio()
        
        # Open audio stream with your USB mic
        stream = audio.open(
            format=FORMAT, 
            channels=CHANNELS,
            rate=RATE, 
            input=True, 
            input_device_index=DEVICE_INDEX,
            frames_per_buffer=CHUNK
        )
        
        print("Recording started... Speak now! (Recording will stop after 2 seconds of silence)")
        frames = []
        
        # Variables to track silence detection
        silent_chunks = 0
        chunks_per_second = RATE / CHUNK  # How many chunks equal one second
        silence_limit = int(silence_duration * chunks_per_second)  # Total chunks for silence duration
        max_chunks = int(max_duration * chunks_per_second)  # Safety limit
        
        chunk_count = 0
        
        # Main recording loop - this is where the intelligence happens
        while chunk_count < max_chunks:
            # Read one chunk of audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            chunk_count += 1
            
            # Calculate the volume level of this chunk
            # We convert bytes to integers and find the maximum amplitude
            import struct
            chunk_data = struct.unpack(f"{CHUNK}h", data)  # 'h' means 16-bit integers
            volume = max(chunk_data) / 32768.0  # Normalize to 0-1 range
            
            # Check if this chunk is "silent" (below our threshold)
            if volume < silence_threshold:
                silent_chunks += 1
                # Print dots to show we're detecting silence (helpful for debugging)
                if silent_chunks % int(chunks_per_second) == 0:  # Print once per second
                    print(".", end="", flush=True)
            else:
                # Reset silence counter when we detect sound
                silent_chunks = 0
            
            # If we've had enough consecutive silent chunks, stop recording
            if silent_chunks >= silence_limit:
                print(f"\nDetected {silence_duration} seconds of silence. Recording complete!")
                break
        
        # Handle the case where we hit the maximum duration
        if chunk_count >= max_chunks:
            print(f"\nReached maximum recording duration of {max_duration} seconds.")
        
        # Clean up audio resources
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save to WAV file
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # Provide feedback about recording quality
        recording_duration = len(frames) / chunks_per_second
        print(f"Recording saved: {recording_duration:.1f} seconds of audio in {output_filename}")
        
        return True
        
    except Exception as e:
        print(f"Error recording audio: {e}")
        return False

def transcribe_audio(audio_filename="user_input.wav"):
    """
    Convert recorded audio to text using Moonshine
    
    Moonshine is optimized for edge devices like Raspberry Pi, making it
    perfect for your use case. It returns a list where the first element
    is the transcribed text.
    
    Args:
        audio_filename (str): Path to the audio file to transcribe
    
    Returns:
        str: Transcribed text, or None if transcription failed
    """
    try:
        # Moonshine returns a list - we want the first (and usually only) result
        transcription_result = moonshine_onnx.transcribe(audio_filename, 'moonshine/tiny')
        
        if transcription_result and len(transcription_result) > 0:
            transcribed_text = transcription_result[0]
            print(f"\n==> User said: '{transcribed_text}'")
            return transcribed_text
        else:
            print("No transcription result received")
            return None
            
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def generate_response(user_input, model="llama3.2:3b"):
    """
    Generate AI response optimized for voice interaction
    
    This function uses your existing prompt engineering approach to ensure
    the response works well when converted to speech. The system context
    guides the model to give concise, conversational responses.
    
    Args:
        user_input (str): The user's question or command
        model (str): Ollama model to use
    
    Returns:
        str: AI response text, or None if generation failed
    """
    try:
        # Your proven system context for voice interactions
        system_context = """
        You are a helpful AI assistant designed for voice interactions. 
        Your responses will be converted to speech and spoken aloud to the user.
    
        Guidelines for your responses:
        - Keep responses conversational and concise (ideally under 50 words)
        - Avoid complex formatting, lists, or visual elements
        - Speak naturally, as if having a friendly conversation
        - If the user's input seems unclear, ask for clarification politely
        - Provide direct answers rather than lengthy explanations unless specifically 
          requested
        """
        
        # Combine system context with user input
        full_prompt = f"{system_context}\n\nUser said: {user_input}\n\nResponse:"
        
        # Generate response using Ollama
        response = ollama.generate(
            model=model,
            prompt=full_prompt
        )
        
        ai_response = response['response'].strip()
        print(f"\n==> AI response: '{ai_response}'\n")
        return ai_response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def text_to_speech(text, output_file="assistant_response.wav"):
    """
    Convert AI response to speech using PIPER
    
    This uses your existing PIPER setup with the Lessac voice model.
    The subprocess approach ensures we can handle any text length
    and get reliable audio output.
    
    Args:
        text (str): Text to convert to speech
        output_file (str): Output WAV file path
    
    Returns:
        bool: True if speech generation successful, False otherwise
    """
    # Path to your voice model
    model_path = "voices/en_US-lessac-low.onnx"
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"Error: Voice model not found at {model_path}")
        return False
    
    try:
        # Run PIPER to generate speech
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
            print(f"Speech generated: {output_file}")
            return True
        else:
            print(f"PIPER error: {stderr}")
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
            print("Audio playback completed")
            return True
        else:
            print(f"Playback error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False

def cleanup_temp_files():
    """
    Clean up temporary audio files
    
    This helps manage disk space on your Raspberry Pi by removing
    the temporary files created during each interaction cycle.
    """
    temp_files = ["user_input.wav", "assistant_response.wav"]
    
    for filename in temp_files:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Cleaned up {filename}")
        except Exception as e:
            print(f"Could not remove {filename}: {e}")

def run_voice_pipeline(cleanup=True):
    """
    Execute the complete voice interaction pipeline with intelligent recording
    
    This is the main orchestration function that ties together all
    the individual components. Think of it as the conductor of your
    audio orchestra - it ensures each section plays at the right time.
    
    The updated pipeline flow is:
    1. Wait for user to press Enter to start recording
    2. Record user's voice with silence detection (stops after 2 seconds of quiet)
    3. Convert speech to text
    4. Generate AI response
    5. Convert response to speech
    6. Play the response
    7. Clean up temporary files
    
    Args:
        cleanup (bool): Whether to remove temporary files after use
    
    Returns:
        bool: True if entire pipeline completed successfully
    """
    print("\n============= Voice AI Pipeline Ready ===========")
    print("  Press [Enter] to start recording your question...")
    
    # Step 1: Wait for user to initiate recording
    input("Press Enter to start recording your question...")
    
    # Step 2: Record user audio with intelligent silence detection
    if not record_audio_with_silence_detection():
        print("Pipeline failed: Could not record audio")
        return False
    
    # Step 3: Transcribe speech to text
    user_text = transcribe_audio()
    if not user_text:
        print("Pipeline failed: Could not transcribe audio")
        return False
    
    # Step 4: Generate AI response
    ai_response = generate_response(user_text)
    if not ai_response:
        print("Pipeline failed: Could not generate response")
        return False
    
    # Step 5: Convert response to speech
    if not text_to_speech(ai_response):
        print("Pipeline failed: Could not generate speech")
        return False
    
    # Step 6: Play the response
    if not play_audio():
        print("Pipeline failed: Could not play audio")
        return False
    
    # Step 7: Clean up temporary files
    if cleanup:
        cleanup_temp_files()
    
    print("\n=== Pipeline Completed Successfully ===\n")
    return True

def continuous_voice_assistant():
    """
    Run the voice assistant in a continuous loop with intelligent recording
    
    This creates a persistent voice assistant that keeps listening
    for new queries. Unlike the previous version, this uses our new
    intelligent recording system that automatically detects when you've
    finished speaking, making conversations feel more natural.
    
    Each interaction cycle:
    1. Waits for you to press Enter
    2. Records until it detects 2 seconds of silence
    3. Processes and responds to your query
    4. Returns to waiting state for next interaction
    """
    print("\nContinuous Voice Assistant started.") 
    print("Each recording will automatically stop after 2 seconds of silence.")
    print("Press Ctrl+C to stop the assistant.\n")
    
    try:
        interaction_count = 0
        while True:
            interaction_count += 1
            print(f"\n--- Interaction #{interaction_count} ---")
            
            success = run_voice_pipeline()
            
            if not success:
                print("There was an issue with this interaction. Let's try again.")
            
            # Brief pause before next interaction
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\nVoice Assistant stopped after {interaction_count} interactions.")
        cleanup_temp_files()

# Example usage and testing functions
if __name__ == "__main__":
    # Uncomment the function you want to test
    
    # Test individual components:
    # detect_audio_devices()  # Use this to verify your USB mic index
    
    # Test single interaction with intelligent recording:
    run_voice_pipeline()
    
    # Run continuous assistant with intelligent recording:
    # continuous_voice_assistant()