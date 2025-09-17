from ultralytics import YOLO
import cv2
import os
import json

# Load YOLO model (nano version)
model = YOLO("yolov8n.pt")

# Path to image or folder
input_path = "images"  # can be a single "test.jpg" or a folder
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

def process_image(image_path):
    results = model(image_path)

    # Get annotated frame
    annotated = results[0].plot()

    # Save output
    filename = os.path.basename(image_path)
    out_path = os.path.join(output_dir, f"annotated_{filename}")
    cv2.imwrite(out_path, annotated)

    detections = []
    # Print detections
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = results[0].names[cls]
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        detections.append({
            "class": name,
            "confidence": conf,
            "bbox": xyxy
        })
        print(f"Detected {name} with confidence {conf:.2f}")
    
    # Save to JSON
    json_out = os.path.join(output_dir, f"detections_{filename}.json")
    with open(json_out, "w") as f:
        json.dump(detections, f, indent=4)

    # Show annotated image
    cv2.imshow("Detections", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Handle single file or folder
if os.path.isdir(input_path):
    for img in os.listdir(input_path):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            process_image(os.path.join(input_path, img))
else:
    process_image(input_path)
