import cv2
import time
import datetime
# import pywhatkit as kit
# Function to check if the current time exceeds the specified timestamp
def is_accident_time(current_time, accident_time):
    return current_time >= accident_time

# Initialize video capture from a camera (you can replace 0 with the camera index or video file path)
cap = cv2.VideoCapture("accident.mp4")

# Initialize background frame for motion detection
background = None

# Parameters for motion detection
motion_threshold = 1000

# Timestamp when the accident occurs (in seconds from the start of the video)
accident_timestamp = 9

# Get the current time
start_time = time.time()

phone_number = "+916397182160"  # Include the country code
message = "Hello, this is a test message!"

# Get the current time
now = datetime.datetime.now()
wait_time_seconds = 10
call_time_seconds = 15
wait_time = time.strftime("%H:%M", time.localtime(time.time() + wait_time_seconds))
call_time = time.strftime("%H:%M", time.localtime(time.time() + call_time_seconds))

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur the frame to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Initialize the background frame
    if background is None:
        background = blurred
        continue

    # Compute the absolute difference between the current frame and the background
    diff = cv2.absdiff(background, blurred)

    # Apply a threshold to the difference to get the foreground mask
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to keep only reasonably sized vehicles
    vehicles = [contour for contour in contours if 50 < cv2.boundingRect(contour)[2] < 150 and 50 < cv2.boundingRect(contour)[3] < 150]

    # Check if it's time to display the "Accident happens" message
    current_time = time.time() - start_time
    if is_accident_time(current_time, accident_timestamp):
        cv2.putText(frame, "Accident happened", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # kit.sendwhatmsg(phone_number, message, int(call_time.split(":")[0]), int(call_time.split(":")[1]),
        #                 int(wait_time.split(":")[0]), int(wait_time.split(":")[1]))
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' key is pressed or video ends
    if cv2.waitKey(30) & 0xFF == ord('q') or not ret:
        break

# Release the video capture object and close windows
cap.release()


cv2.destroyAllWindows()
