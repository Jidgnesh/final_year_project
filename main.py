import cv2
import numpy as np
import time
from twilio.rest import Client

# Set up Twilio client for SMS notification
TWILIO_SID = ''
TWILIO_AUTH_TOKEN = ''
TWILIO_PHONE_NUMBER = ''
EMERGENCY_PHONE_NUMBER = '+'
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Set up OpenCV video capture
cap = cv2.VideoCapture('v13.mp4')

# Define a function to send an SMS notification using Twilio
def send_sms_notification():
    message = client.messages.create(
        body='Vehicle accident detected. Please send emergency services.',#.format(latitude, longitude),
        from_=TWILIO_PHONE_NUMBER,
        to=EMERGENCY_PHONE_NUMBER
    )
    print('SMS notification sent.')

# Initialize background subtraction algorithm
fgbg = cv2.createBackgroundSubtractorMOG2()

# Flag to indicate if an accident has been detected
accident_detected = False

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction to extract moving objects
    fgmask = fgbg.apply(frame)
    
    # Apply thresholding to the foreground mask to create a binary image
    thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]
    
    # Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over the contours to check if any of them correspond to a vehicle accident
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # adjust the threshold as needed
            # Set the accident_detected flag to True
            accident_detected = True
            send_sms_notification()
            
    # Display the video feed with the detected contours
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Accident Detection', frame)
    
    # Stop processing frames if an accident has been detected
    if accident_detected:
        time.sleep(20)  # wait for 20 seconds before sending another notification
        accident_detected = False
    
    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
