import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# Load a pre-trained DeepLab model
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model.eval()

# Setup transformation for the input image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((480, 640)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera is not accessible")
    exit()

def segment_and_display(frame):
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Create a color mask
    print(output_predictions.max())
    colors = np.array([
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (0, 64, 64), (64, 0, 64), (192, 192, 192)
]) 
    color_mask = colors[output_predictions.cpu().numpy()]  # Apply colors to the predictions

    # Resize color_mask to match frame dimensions
    color_mask = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to a numpy array and overlay it
    color_mask = color_mask.astype('uint8')
    frame_with_mask = cv2.addWeighted(frame, 0.5, color_mask, 0.5, 0)

    cv2.imshow('Segmented Video', frame_with_mask)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        segment_and_display(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
