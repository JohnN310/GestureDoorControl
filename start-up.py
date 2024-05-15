import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Initialize the hand detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Open the default camera
cap = cv2.VideoCapture(0)

# Define the hand gestures for controlling the door
gestures = {
    "Open": [0, 1, 1, 0, 0],  # Only thumb is up
    "Close": [1, 1, 0, 0, 0]  # Only index and middle fingers are up
}

door_state = "Closed"

def detect_gesture(finger_status):
    for gesture, pattern in gestures.items():
        if finger_status == pattern:
            return gesture
    return "Unknown"

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
        break
    
    # Detect hands and find the landmarks
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        bbox = hand["bbox"]
        center = hand["center"]
        handType = hand["type"]
        
        # Determine which fingers are up
        fingers = detector.fingersUp(hand)
        
        # Detect gesture based on finger status
        gesture = detect_gesture(fingers)
        
        # Control the door based on the detected gesture
        if gesture == "Open" and door_state == "Closed":
            door_state = "Open"
            print("Door is now Open.")
        elif gesture == "Close" and door_state == "Open":
            door_state = "Closed"
            print("Door is now Closed.")
        
        # Display the gesture and door state on the image
        cv2.putText(img, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f'Door: {door_state}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the flipped image
    cv2.imshow("Hand Gesture Control for Door (Flipped)", img)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
