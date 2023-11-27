import cv2
import torch
import torchvision
import torchvision.transforms as T

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load a class list for the model (adjust class labels as needed)
class_labels = ['background', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']

# Define a function to perform object detection
def detect_objects(frame):
    # Convert the frame to a tensor
    frame = T.ToTensor()(frame)
    frame = frame.unsqueeze(0)

    with torch.no_grad():
        # Perform object detection
        predictions = model(frame)

    # Extract detected objects, their bounding boxes, and labels
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']

    return boxes, labels

# Open the video file
cap = cv2.VideoCapture('video.mp4')

# Set minimum object area for small object filtering
min_object_area = 50  # Adjust as needed

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection on the frame
    boxes, labels = detect_objects(frame)

    # Draw bounding boxes on the frame for detected cars, skipping small objects
    for box, label in zip(boxes, labels):
        if label < len(class_labels) and class_labels[label] == 'car':
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            object_area = w * h

            # Skip small objects
            if object_area < min_object_area:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, class_labels[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame with car detection
    cv2.imshow('Video with Car Detection', frame)

    # Press 'q' to exit the video loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
