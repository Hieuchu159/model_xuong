from ultralytics import YOLO

# Load the exported TFLite model
tflite_model = YOLO("/content/best_saved_model/best_float16.tflite")

# Run inference
results = tflite_model("/content/a.jpg")