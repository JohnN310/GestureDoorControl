import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np


#variables
faceDetectedCounter = 0
faceOrNotFaceDetected = False
totalCounter = 30

# Initialize the hand detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Load the homeowner's face
homeowner_img = cv2.imread('C://Users//anhkh//OneDrive//GitHub//Door Control using Hand Gestures//GestureDoorControl//homeowner.JPG')
homeowner_gray = cv2.cvtColor(homeowner_img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to check if the face matches the homeowner's face
def is_homeowner(face_gray):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(homeowner_gray, None)
    kp2, des2 = sift.detectAndCompute(face_gray, None)
     # If descriptors are not found, return False
    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    print(len(matches))
    return len(matches) > 70  # Assuming at least 10 matches indicate a match

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

    if not hands:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_detected = False
        for (x, y, w, h) in faces:
            if faceOrNotFaceDetected is False:
                face_gray = gray[y:y+h, x:x+w]
                face_detected = is_homeowner(face_gray)
                if face_detected:
                    print("Homeowner!!")
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, "Homeowner", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    faceOrNotFaceDetected = True
                else:
                    print("Unknown!!")
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    faceOrNotFaceDetected = True
            

        if faceOrNotFaceDetected:
            faceDetectedCounter += 1
            if faceDetectedCounter > totalCounter:
                faceDetectedCounter = 0
                faceOrNotFaceDetected = False
    
    else:
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
    cv2.imshow("Hand Gesture Control for Door", img)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
