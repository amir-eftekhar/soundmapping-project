import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
import threading
from queue import Queue
import pygame
from directional_aud import play_directional_sound

# Load pre-trained MiDaS model
model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
model.eval()

# Check for GPU availability and move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare transformations for the input image
transform = Compose([
    Resize((256, 256)),  # Reduced size for faster processing
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_depth(frame, transform, model, device, result_queue):
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Predict the depth map
    with torch.no_grad():
        depth = model(input_tensor)
    depth_map = depth.squeeze().cpu().numpy()
    
    result_queue.put(depth_map)

def play_audio_based_on_depth(depth_map):
    # Normalize depth map to range between 0 and 1
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

    # Calculate the middle part of the depth map
    height, width = depth_map.shape
    middle_section = depth_map_normalized[height // 2 - 50:height // 2 + 50, width // 2 - 50:width // 2 + 50]

    # Calculate the average depth in the middle section
    average_depth = np.mean(middle_section)
    volume = 1 - average_depth  # Closer objects are louder
    pan = 0  # No panning, centered audio

    frequency = 440 + (1 - average_depth) * 440  # Frequency from 440 Hz to 880 Hz

    # Play the sound
    play_directional_sound(frequency, 1, volume, pan)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera is not accessible")
    exit()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    result_queue = Queue()
    
    # Predict the depth map in a separate thread
    depth_thread = threading.Thread(target=predict_depth, args=(frame, transform, model, device, result_queue))
    depth_thread.start()
    depth_thread.join()
    
    # Retrieve the depth map from the queue
    depth_map = result_queue.get()

    # Resize depth map to match the frame size
    depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # Normalize the depth map for better visualization
    depth_map_normalized = cv2.normalize(depth_map_resized, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_MAGMA)

    # Play audio based on the depth map
    play_audio_based_on_depth(depth_map_resized)

    # Display the results
    cv2.imshow('Original', frame)
    cv2.imshow('Depth Map', depth_map_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
pygame.quit()
