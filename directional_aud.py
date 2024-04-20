import numpy as np
from scipy.io.wavfile import write
import simpleaudio as sa

def play_directional_sound(frequency, duration, volume, pan, fs=44100):
    
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * frequency * t) * volume
    waveform_integers = np.int16(waveform * 32767)

    
    stereo_waveform = np.zeros((len(waveform_integers), 2), dtype=np.int16)
    stereo_waveform[:, 0] = waveform_integers * (1 - pan if pan < 0 else 1)  
    stereo_waveform[:, 1] = waveform_integers * (1 + pan if pan > 0 else 1)  

    # Play the sound
    play_obj = sa.play_buffer(stereo_waveform, 2, 2, fs)
    play_obj.wait_done()


frequency = 440  
duration = 1  
volume = 0.5  
pan = -0.5  

play_directional_sound(frequency, duration, volume, pan)





