print("working")
import cv2
import numpy as np
from threading import Thread, Semaphore
from directional_aud import play_directional_sound
from trial1 import sound_mapping
# Initialize camera
print("working")
cap = cv2.VideoCapture(0)
print("Camera initialized...")
semaphore = Semaphore(10)

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

# Detection function
def detect_objects(img):
    semaphore.acquire()
    try:
        print(f"Starting sound for {label}...")
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

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    print("working")
    try:
        indexes, boxes, class_ids, confidences = detect_objects(frame)
        print("Objects detected")
    except Exception as e:
        print(f"Error in detect_objects: {e}")
        continue

    try:
        sound_mapping(indexes, boxes, class_ids, frame.shape[1])
        print("Sound mapped")
    except Exception as e:
        print(f"Error in sound_mapping: {e}")
        continue

    try:
        display_frame(frame, indexes, boxes, class_ids, confidences)
        print("Frame displayed")
    except Exception as e:
        print(f"Error in display_frame: {e}")
        continue

    cv2.imshow('sound mapping', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break
    

    # Play directional sound for each detected object
    for i in indexes:
        x, y, w, h = boxes[i[0]]
        label = str(classes[class_ids[i[0]]])
        Thread(target=play_directional_sound, args=(label, x, w, frame.shape[1])).start()

    

cap.release()
cv2.destroyAllWindows()