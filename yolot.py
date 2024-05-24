import cv2
import torch
import numpy as np

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)

# Setup stereo vision
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Load images
frame_left = cv2.imread('left_image.jpg')  # Replace with your image path
frame_right = cv2.imread('right_image.jpg')  # Replace with your image path

# Object detection on left image
results = model(frame_left)
detections = results.pandas().xyxy[0]  # Extract bounding boxes

# Compute disparity and depth map
disparity = stereo.compute(cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY))
depth = cv2.reprojectImageTo3D(disparity, Q)  # Q is the disparity-to-depth mapping matrix

# Map detected objects to depth information
for index, row in detections.iterrows():
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    object_depth = depth[y1:y2, x1:x2]

    # Calculate average or minimum depth within the bounding box
    average_depth = np.nanmean(object_depth[object_depth != np.inf])
    print(f"Detected {row['name']} at an average depth of {average_depth:.2f} meters")

    # Draw bounding box and depth information on the image
    cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame_left, f"{row['name']}: {average_depth:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Object Detection', frame_left)

# Display the depth map
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
cv2.imshow('Depth Map', depth_colormap)

cv2.waitKey(0)
cv2.destroyAllWindows()