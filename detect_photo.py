from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model (nano version, fast and small)
model = YOLO("yolov8n.pt")

# Load your photo
image_path = "test.jpg"  # replace with your photo path
results = model(image_path)

# Display detections
annotated = results[0].plot()
cv2.imshow("Detected Items", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
