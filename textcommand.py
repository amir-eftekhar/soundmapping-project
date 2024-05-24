import speech_recognition as sr
from speak import speak
def listen_and_recognize():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  
        print("Please say something...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

def lar_w_command_v1(command):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  
        while True:  # Loop forever
            print("Listening for activation command...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                words = text.lower().split()
                if command in words:
                    print("Activation command heard, processing following speech...")
                    audio = recognizer.listen(source)
                    try:
                        text = recognizer.recognize_google(audio)
                        print(f"You said: {text}")
                        speak(text)
                        break  # Stop listening after the speech has stopped
                    except sr.UnknownValueError:
                        print("Google Speech Recognition could not understand the audio")
                    except sr.RequestError as e:
                        print(f"Could not request results from Google Speech Recognition service; {e}")
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
def lar_w_command(command):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  
        while True:  # Loop forever
            print("Listening for activation command...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                words = text.lower().split()
                if command in words:
                    print(f"Activation command '{command}' heard. Listening for the phrase...")
                    audio = recognizer.listen(source)
                    try:
                        text = recognizer.recognize_google(audio)
                        print(f"You said: {text}")
                        return text  # Return the phrase heard after the activation command
                    except sr.UnknownValueError:
                        print("Google Speech Recognition could not understand the audio")
                    except sr.RequestError as e:
                        print(f"Could not request results from Google Speech Recognition service; {e}")
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
if __name__ == "__main__":
    
    lar_w_command("hello" )