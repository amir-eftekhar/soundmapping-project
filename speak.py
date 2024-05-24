from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import pygame
import tempfile
'''def speak(text):
    tts = gTTS(text=text, lang='en', slow=False)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)  # Go to the beginning of the file-like object
    # Load this file-like object into a pydub AudioSegment
    audio = AudioSegment.from_file(fp, format="mp3")
    # Play the audio
    play(audio)'''
    


def speak(text):
    tts = gTTS(text=text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(fp.name)
        pygame.mixer.init()
        pygame.mixer.music.load(fp.name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
