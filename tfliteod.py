import cv2
import numpy as np
import tensorflow as tf
from tflite_support import metadata

model_file = '1.tflite'  # Path to your TFLITE model file
displayer = metadata.MetadataDisplayer.with_model_file(model_file)

# Extract and print the associated file (labels file)
'''associated_files = displayer.get_associated_files()
labels_file = associated_files[0].name
print("Labels file:", labels_file)
labels_content = associated_files[0].content.decode("utf-8")
print("Labels content:\n", labels_content)
# Load TFLite model and allocate tensors.'''
interpreter = tf.lite.Interpreter(model_path="1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video file or capture from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image to required size and cast
    input_shape = input_details[0]['shape']
    input_data = np.expand_dims(cv2.resize(frame, (input_shape[1], input_shape[2])), axis=0)
    input_data = input_data.astype(np.uint8)

    # Point the data to be used for testing and run the interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    labels = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
        'hair drier', 'toothbrush'
    ]

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if scores[i] > 0.5:
            box = boxes[i]
            # Draw rectangle around the object
            cv2.rectangle(frame, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 0), 2)
            
            # Display label and confidence score
            label = f"{labels[int(classes[i])]}: {scores[i]*100:.2f}%"
            y = box[0] - 15 if box[0] > 20 else box[0] + 15
            cv2.putText(frame, label, (int(box[1]), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Object detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()