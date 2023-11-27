import cv2
import numpy as np

# Initialize video capture from a webcam or video file
cap = cv2.VideoCapture('vi.mp4')  # Use 0 for default webcam, or provide a video file path

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Parameters for collision detection
min_collision_distance = 100  # Adjust as needed

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction to detect moving objects
    fgmask = fgbg.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Flag to track whether a collision occurred
    collision_detected = False

    # Variables to store information about the closest pair
    closest_distance = float('inf')
    closest_pair = None

    # Check for collisions between objects
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            x1, y1, w1, h1 = cv2.boundingRect(contours[i])
            x2, y2, w2, h2 = cv2.boundingRect(contours[j])
            center1 = (x1 + w1 // 2, y1 + h1 // 2)
            center2 = (x2 + w2 // 2, y2 + h2 // 2)
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

            if distance < min_collision_distance and distance < closest_distance:
                closest_distance = distance
                closest_pair = (i, j)
                collision_detected = True

    # Show the "Collision Warning!" message only if a collision is detected
    if collision_detected:
        cv2.putText(frame, "Collision Warning!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Optionally, you can highlight the closest pair of objects by drawing rectangles
        i, j = closest_pair
        x1, y1, w1, h1 = cv2.boundingRect(contours[i])
        x2, y2, w2, h2 = cv2.boundingRect(contours[j])
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Collision Detection', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Press Esc to exit
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
