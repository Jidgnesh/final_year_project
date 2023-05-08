import cv2
import numpy as np
import time
import requests
from twilio.rest import Client
from geopy.geocoders import Nominatim

# Set up Twilio client for SMS notification
TWILIO_SID = ''
TWILIO_AUTH_TOKEN = ''
TWILIO_PHONE_NUMBER = ''
EMERGENCY_PHONE_NUMBER = ''
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Set up OpenCV video capture
cap = cv2.VideoCapture('car_1.mp4')

# Initialize background subtraction algorithm
fgbg = cv2.createBackgroundSubtractorMOG2()

# Set up geocoder
geolocator = Nominatim(user_agent='accident_detector')

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
            # Convert the coordinates of the center of the contour to latitude and longitude
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            lat, lon = geolocator.reverse(f"{cy}, {cx}").point[:2]
            location = geolocator.reverse(f"{lat}, {lon}")
            address = location.address
            
            # Send an SMS notification to emergency services with the location of the accident
            message = client.messages.create(
                body=f"Vehicle accident detected at {address}. Please send emergency services.",
                from_=TWILIO_PHONE_NUMBER,
                to=EMERGENCY_PHONE_NUMBER
            )
            print('SMS notification sent.')
            
            time.sleep(10)  # wait for 10 seconds before sending another notification
            
    # Display the video feed with the detected contours
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Accident Detection', frame)
    
    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
