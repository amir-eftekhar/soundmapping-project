print("working")
import cv2
import numpy as np
from threading import Thread, Semaphore
from directional_aud import play_directional_sound
import threading
import queue
# Initialize camera
print("working")
cap = cv2.VideoCapture(0)
print("Camera initialized...")
semaphore = Semaphore(10)
q = queue.Queue()
if not cap.isOpened():
    print("Error: Camera is not accessible")
    exit()
# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

def sound_mapping(indexes, boxes, class_ids, width, classes):
    threads = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            distance = width // (w + 1)  # Simple distance estimation
            label = str(classes[class_ids[i]])

            # Calculate pan based on object position
            pan = (x + w / 2 - width / 2) / (width / 2)  # Pan range -1 to 1

            # Calculate volume based on distance, smaller width of the box means farther away
            volume = min(1, 2 / (distance if distance > 0 else 1))

            # Play sound with directional audio
            thread = threading.Thread(target=play_directional_sound, args=(440, 1, volume, pan))
            threads.append(thread)
            thread.start()
    return threads
# Detection function
def detect_objects(img):
    semaphore.acquire()
    boxes = []
    try:
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        print("Starting object detection...")
        outs = net.forward(output_layers)
        print("Object detection completed.")
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        q.put((indexes, boxes, class_ids, confidences))  # Put the result into the queue
    except Exception as e:
        print(f"Error in play_directional_sound: {e}")
    finally:

        semaphore.release()
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indexes, boxes, class_ids, confidences


# Display function
def display_frame(frame, indexes, boxes, class_ids, confidences):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

threads = []
# Main loop

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Start a new thread for object detection
    detect_thread = threading.Thread(target=detect_objects, args=(frame,))
    detect_thread.start()

    # Wait for the detection thread to finish and get the results
    detect_thread.join()
    indexes, boxes, class_ids, confidences = q.get()  # Get the result from the queue
    

    # Start a new thread for sound mapping
    sound_thread = threading.Thread(target=sound_mapping, args=(indexes, boxes, class_ids, frame.shape[1], classes))
    sound_thread.start()

    # Wait for the sound mapping thread to finish
    sound_thread.join()

    try:
        display_frame(frame, indexes, boxes, class_ids, confidences)
        print("Frame displayed")
    except Exception as e:
        print(f"Error in display_frame: {e}")
        continue

    cv2.imshow('sound mapping', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

# Join all threads before exiting
for thread in threads:
    thread.join()

cap.release()
cv2.destroyAllWindows()