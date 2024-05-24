import cv2
import speech_recognition as sr
import nltk
import numpy as np
import os
from queue import Queue
from threading import Thread

# Ensure that you have downloaded the necessary nltk packages
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize video capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def speak(text):
    """
    Convert text to speech.
    """
    print(text)  # Simulating speech with print statements for demonstration

def detect_objects(frame, output_queue):
    """
    Dummy implementation of object detection, replace with actual object detection code.
    """
    # Simulated outputs
    indexes = [0]
    boxes = [(100, 100, 50, 50)]  # Example box coordinates
    class_ids = [1]  # Example class id
    confidences = [0.8]  # Example confidence

    # Put the simulated detection results into the queue
    output_queue.put((indexes, boxes, class_ids, confidences))

def listen_for_command():
    """
    Listen for a voice command and convert it to text.
    """
    try:
        with sr.Microphone() as source:
            print("Listening for command...")
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio)
            print(f"Command: {command}")
            return command
    except sr.UnknownValueError:
        print("Sorry, I did not understand that. Please repeat your command.")
        return None

def main():
    while True:
        command = listen_for_command()
        if command:
            tokens = nltk.word_tokenize(command)
            tagged = nltk.pos_tag(tokens)
            objects = [word for word, pos in tagged if pos == 'NN']

            ret, frame = cap.read()
            if not ret:
                speak("Failed to grab frame")
                break

            output_queue = Queue()
            detect_thread = Thread(target=detect_objects, args=(frame, output_queue))
            detect_thread.start()
            detect_thread.join()

            if not output_queue.empty():
                indexes, boxes, class_ids, confidences = output_queue.get()

                found = False
                for i in range(len(boxes)):
                    if i in indexes:
                        label = str(classes[class_ids[i]])
                        if label in objects:
                            box = boxes[i]
                            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                            found = True
                            speak(f"Found a {label}")

                if not found:
                    speak("Object not found, please try again.")

                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
