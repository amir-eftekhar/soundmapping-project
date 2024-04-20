import cv2
import numpy as np
import threading
from directional_aud import play_directional_sound

# Initialize camera
cap = cv2.VideoCapture(0)

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
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

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
    return indexes, boxes, class_ids, confidences

# Sound mapping
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

# Main loop
'''while True:
    ret, frame = cap.read()
    if ret:
        indexes, boxes, class_ids, confidences = detect_objects(frame)
        sound_mapping(indexes, boxes, class_ids, frame.shape[1])

    key = cv2.waitKey(1)
    if key == 27:
        break'''

cap.release()
cv2.destroyAllWindows()
