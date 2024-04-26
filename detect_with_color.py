import cv2
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn
from PIL import Image
from torchvision import transforms

class WallDetector(nn.Module):
    def __init__(self):
        super(WallDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*64*64, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 2)  # The output is a 2D vector representing the distance to the wall

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
    
def detect_wall_w_t(image_path):
    # Load the trained model
    model = WallDetector()
    model.load_state_dict(torch.load('wall_detector.pth'))
    model.eval()

    # Load the image
    image = Image.open(image_path)

    # Apply the same transformations as during training
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize the image to 128x128 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    image = transform(image).unsqueeze(0)  # Add an extra dimension for the batch size

    # Use the model to predict the distance to the wall
    distance = model(image)

    return distance
def detect_wall_w_c(image):
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Perform k-means clustering to segment the image into k color clusters
    kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters based on your requirements
    kmeans.fit(pixels)
    
    # Find the largest cluster
    counts = np.bincount(kmeans.labels_)
    largest_cluster = kmeans.cluster_centers_[np.argmax(counts)]
    
    # Convert the largest cluster color to the HSV color space
    largest_cluster_hsv = cv2.cvtColor(np.uint8([[largest_cluster]]), cv2.COLOR_BGR2HSV)[0, 0]
    
    # Define a color range around the largest cluster color
    lower_color = np.clip(largest_cluster_hsv - 20, 0, 255)  # Adjust these values based on your requirements
    upper_color = np.clip(largest_cluster_hsv + 20, 0, 255)  # Adjust these values based on your requirements
    
    # Create a mask of pixels within the color range
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a binary image where the largest color cluster is blue and everything else is black
    binary = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Only consider contours with area greater than 100
            cv2.drawContours(binary, [contour], -1, (255, 0, 0), thickness=cv2.FILLED)
    
    # Overlay the binary image on the original image to create the highlight effect
    output = cv2.addWeighted(image, 0.7, binary, 0.3, 0)
    
    return output