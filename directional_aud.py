import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import pyaudio
import sounddevice as sd
import time
import simpleaudio as sa

import pygame

def play_directional_sound_v1(frequency, duration, volume, pan, fs=44100):
    try: 
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        waveform = np.sin(2 * np.pi * frequency * t) * volume
        waveform_integers = np.int16(waveform * 32767)

        stereo_waveform = np.zeros((len(waveform_integers), 2), dtype=np.int16)
        stereo_waveform[:, 0] = waveform_integers * (1 - pan if pan < 0 else 1)  
        stereo_waveform[:, 1] = waveform_integers * (1 + pan if pan > 0 else 1)  

        # Play the sound
        sd.play(stereo_waveform, samplerate=fs)
        sd.wait()
    except Exception as e:
        print(f"An error occurred: {e}")

def play_directional_sound_v2(frequency, duration, volume, pan, fs=44100):
    try: 
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        waveform = np.sin(2 * np.pi * frequency * t) * volume
        waveform *= np.hanning(len(waveform))  # Apply a Hanning window
        waveform_integers = np.int16(waveform * 32767)

        stereo_waveform = np.zeros((len(waveform_integers), 2), dtype=np.int16)
        stereo_waveform[:, 0] = waveform_integers * (1 - pan if pan < 0 else 1)  
        stereo_waveform[:, 1] = waveform_integers * (1 + pan if pan > 0 else 1)  

        # Play the sound
        sd.play(stereo_waveform, samplerate=fs)
        sd.wait()
    except Exception as e:
        print(f"An error occurred: {e}")
        


def play_directional_sound(frequency, duration, volume, pan, fs=44100):
    try:
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        waveform = np.sin(2 * np.pi * frequency * t) * volume
        waveform *= np.hanning(len(waveform))  # Apply a Hanning window
        waveform_integers = np.int16(waveform * 32767)

        stereo_waveform = np.zeros((len(waveform_integers), 2), dtype=np.int16)
        stereo_waveform[:, 0] = waveform_integers * (1 - pan if pan < 0 else 1)
        stereo_waveform[:, 1] = waveform_integers * (1 + pan if pan > 0 else 1)

        # Initialize Pygame mixer
        pygame.mixer.init(frequency=fs, size=-16, channels=2)
        sound = pygame.sndarray.make_sound(stereo_waveform)
        sound.set_volume(volume)
        sound.play()
        pygame.time.wait(int(duration * 1000))
    except Exception as e:
        print(f"An error occurred: {e}")

'''
frequency = 440  
duration = 1  
volume = 0.5  
pan = -0.5  

play_directional_sound(frequency, duration, volume, pan)

def play_moving_sound(frequency, duration, volume, start_pan, end_pan, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * frequency * t)
    waveform_integers = np.int16(waveform * 32767)

    # Increase the pan range
    pan_values = np.sin(np.linspace(-np.pi / 2, np.pi / 2, len(t))) * (end_pan - start_pan) * 2 / 2 + (end_pan + start_pan) / 2

    # Add a delay between the ears
    delay_samples = int(fs * 0.001)  # 1 ms delay
    delayed_waveform = np.roll(waveform_integers, delay_samples)

    stereo_waveform = np.zeros((len(waveform_integers), 2), dtype=np.int16)
    for i, pan in enumerate(pan_values):
        stereo_waveform[i, 0] = delayed_waveform[i] * (1 - pan if pan < 0 else 1)
        stereo_waveform[i, 1] = waveform_integers[i] * (1 + pan if pan > 0 else 1)

    sd.play(stereo_waveform, samplerate=fs)
    sd.wait()
def play_spatial_audio(frequency=440, duration=1, volume=0.5, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * frequency * t)
    waveform_integers = np.int16(waveform * 32767 * volume)

    # Create a stereo waveform with two channels
    stereo_waveform = np.zeros((len(waveform_integers), 2), dtype=np.int16)

    # Generate a sequence of pan values that moves the sound around the listener's head
    pan_values = np.sin(2 * np.pi * t / duration)  # One full cycle per second

    # Apply the pan values to the stereo waveform
    for i, pan in enumerate(pan_values):
        stereo_waveform[i, 0] = waveform_integers[i] * (1 - pan)
        stereo_waveform[i, 1] = waveform_integers[i] * (1 + pan)

    sd.play(stereo_waveform, samplerate=fs)
    sd.wait()
play_spatial_audio()

play = False

if play: 
    def play_left_to_right(frequency, duration, volume):
        play_moving_sound(frequency, duration, volume, -1, 1)
        time.sleep(1)  # Pause for 1 second

    def play_right_to_left(frequency, duration, volume):
        play_moving_sound(frequency, duration, volume, 1, -1)
        time.sleep(1)  # Pause for 1 second

    def play_front_to_back(frequency, duration, volume):
        play_moving_sound(frequency, duration, volume, 0, 1)
        time.sleep(1)  # Pause for 1 second

    def play_back_to_front(frequency, duration, volume):
        play_moving_sound(frequency, duration, volume, 1, 0)
        time.sleep(1)  # Pause for 1 second
    play_left_to_right(440, 2, 0.5)
    time.sleep(1)
    play_right_to_left(440, 2, 0.5)
    time.sleep(1)
    play_front_to_back(440, 2, 0.5)
    time.sleep(1)
    play_back_to_front(440, 2, 0.5)'''