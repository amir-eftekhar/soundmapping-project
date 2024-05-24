import numpy as np
import sounddevice as sd
import time
from scipy.signal import chirp, find_peaks
from speakert import switch_audio_output, get_audio_devices, get_current_output_device

def generate_sound(frequency=21000, duration=1.0, fs=44100):
    """
    Generate a sine wave at a specified frequency and duration.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * frequency * t)
    return waveform * (2**15 - 1)

def record_sound(duration=1.0, fs=44100):
    """
    Record audio for a given duration and sampling rate.
    """
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    return recording.flatten()

def play_and_record(waveform, fs=44100):
    """
    Play a sound and simultaneously record the microphone input.
    """
    recording = sd.playrec(waveform, samplerate=fs, channels=1)
    sd.wait()  # Wait until audio processing is finished
    return recording.flatten()


fs = 44100  # Sampling frequency
duration = 1  # Duration in seconds

# Save the current default output device
default_device = get_current_output_device()
print(f"Default output device: {default_device}")

# Get audio devices and set to a speaker that can handle ultrasonic frequencies
devices = get_audio_devices()
print(f"Available audio devices: {devices}")
for device in devices:
    if 'speakers' in device.lower():
        switch_audio_output(device)
        print(f"Switched output to: {device}")
        break

# Generate and play ultrasonic sound
ultrasonic_wave = generate_sound(frequency=21000, duration=duration, fs=fs)
print(f"Generated ultrasonic wave with {len(ultrasonic_wave)} samples")

echo = play_and_record(ultrasonic_wave, fs=fs)
print(f"Recorded echo with {len(echo)} samples")

# Switch back to original default speaker
switch_audio_output(default_device)
print(f"Switched output back to default device: {default_device}")

peaks, _ = find_peaks(echo, height=0.2)  # Find peaks in the recorded echo
print(f"Found {len(peaks)} peaks in the echo")

for peak in peaks:
    time_delay = peak / fs
    distance = time_delay * 343.2 / 2  # speed of sound = 343.2 m/s at 20 degrees Celsius
    print(f"Detected echo at {time_delay*1000:.2f} ms, estimated distance: {distance:.2f} meters")