print("true")
import cv2
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera is not accessible")
    exit()

while True:
    print("working")
    ret, frame = cap.read()
    cv2.imshow('Camera Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()