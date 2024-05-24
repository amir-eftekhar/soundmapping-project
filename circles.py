import sounddevice as sd
import numpy as np
import time

def rotating_tone_stereo(frequency=440, duration=10, samplerate=44100):
    """Generate a tone that appears to rotate around the listener using stereo."""
    total_samples = int(duration * samplerate)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    
    # Generate a sine wave for the tone
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Initialize a 2-channel audio buffer
    buffer = np.zeros((total_samples, 2))
    
    # Calculate angle of rotation per sample (full circle over duration)
    angle_step = 2 * np.pi / total_samples
    
    for i in range(total_samples):
        angle = i * angle_step
        # Amplitude modulation for stereo effect
        buffer[i, 0] = tone[i] * (0.5 * np.sin(angle) + 0.5)  # Left channel
        buffer[i, 1] = tone[i] * (0.5 * np.cos(angle) + 0.5)  # Right channel
    
    return buffer

# Set the parameters for the audio stream
samplerate = 44100  # Audio sampling rate
channels = 2        # Number of audio channels (stereo)

# Create the rotating tone audio buffer
audio_data = rotating_tone_stereo()

# Define a callback function that plays the generated audio
def callback(outdata, frames, time, status):
    global start_idx, audio_data
    end_idx = start_idx + frames
    outdata[:] = audio_data[start_idx:end_idx] if end_idx < len(audio_data) else np.zeros((frames, channels))
    start_idx = 0 if end_idx >= len(audio_data) else end_idx

start_idx = 0

# Start an OutputStream with the callback
with sd.OutputStream(samplerate=samplerate, channels=channels, callback=callback, dtype='float32'):
    input("Press Enter to stop playback...")  # Keeps the stream open
