import subprocess
import sys
import numpy as np
from scipy.io.wavfile import write
import simpleaudio as sa
from directional_aud import play_directional_sound

def get_audio_devices():
    result = subprocess.run(["SwitchAudioSource", "-a"], capture_output=True, text=True)
    if result.stderr:
        print("Error fetching devices:", result.stderr)
    devices = result.stdout.split('\n')
    return [device.strip() for device in devices if device.strip()]

def get_current_output_device():
    result = subprocess.run(["SwitchAudioSource", "-c"], capture_output=True, text=True)
    if result.stderr:
        print("Error getting current device:", result.stderr)
    return result.stdout.strip()

def switch_audio_output(device_name):
    print(f"Attempting to switch to {device_name}")
    result = subprocess.run(["SwitchAudioSource", "-s", device_name], capture_output=True, text=True)
    if result.stderr:
        print("Error switching device:", result.stderr)
    print("SwitchAudioSource output:", result.stdout)

if __name__ == "__main__":
    default_device = get_current_output_device()
    print(f"Default device: {default_device}")

    devices = get_audio_devices()
    print(f"Available devices: {devices}")

    target_device = next((device for device in devices if 'speaker' in device.lower()), None)
    if target_device:
        print(f"Switching to {target_device} and playing sound...")
        switch_audio_output(target_device)
        play_directional_sound(440, 1, 0.5, 0.5)
    else:
        print("No suitable 'speaker' device found.")

    print(f"Switching back to {default_device}...")
    switch_audio_output(default_device)
