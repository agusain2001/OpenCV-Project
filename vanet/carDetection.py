import cv2
import numpy as np

# Define the minimum width and height for detected objects (cars)
min_width_react = 80
min_height_react = 80   

# Open the video file
cap = cv2.VideoCapture('video.mp4')

# Create a background subtractor using MOG2
algo = cv2.createBackgroundSubtractorMOG2()

# Set the desired FPS (e.g., 10 FPS)
desired_fps = 10

# Calculate the delay in milliseconds based on the desired FPS
frame_delay = int(1000 / desired_fps)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for background subtraction
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Apply background subtraction
    img_sub = algo.apply(blur)

    # Dilate the resulting image to enhance object boundaries
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))

    # Apply morphological operations for further noise reduction
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernal)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernal)

    # Find contours in the processed image
    counterShaper, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in counterShaper:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(c)

        # Check if the width and height meet the minimum criteria for a car
        if w >= min_width_react and h >= min_height_react:
            # Draw a rectangle around the detected car
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the original frame with car detection
    cv2.imshow('Video with Car Detection', frame)

    # Press 'q' to exit the video loop
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
