from ultralytics import YOLO

# Load a model
model = YOLO('MultiSpeech/FaceDetector/models/yolov8n.pt')  # You can replace 'yolov8n.pt' with any other model version like 'yolov8s.pt', 'yolov8l.pt', etc.

# Print the model architecture
print(model)