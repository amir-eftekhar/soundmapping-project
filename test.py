import sounddevice as sd
import numpy as np

def test_sound(duration=1.0, fs=44100):
    """Test sound generation and playback."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * 440 * t)  # A simple 440 Hz tone
    sd.play(waveform, samplerate=fs)
    sd.wait()

test_sound()