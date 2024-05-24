import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained MiDaS model
model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
model.eval()

# Prepare transformations for the input image
transform = Compose([
    Resize((512, 512)),  # Increase size for more detail
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Using OpenCV to capture images from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the captured frame to PIL Image and apply transformation
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(input_image).unsqueeze(0)
    
    # Predict the depth map
    with torch.no_grad():
        depth = model(input_tensor)
    
    # Convert depth to a displayable format
    depth_image = depth.squeeze().numpy()
    depth_image = depth_image / (depth_image.max() / 2)  # Increase sensitivity
    
    # Display the result
    plt.imshow(depth_image, cmap='plasma')
    plt.pause(1)  # Update every second

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()