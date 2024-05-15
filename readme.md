# Hand Gesture Controlled Smart Door System

This project implements a hand gesture-controlled smart door system using Python, OpenCV, and a hand tracking module. The system allows users to control the opening and closing of a door through predefined hand gestures detected by a camera.

## Features

- **Hand Gesture Detection**: Utilizes computer vision techniques to detect and recognize hand gestures in real-time.
- **Door Control**: Allows users to open and close a virtual door by performing specific hand gestures.
- **Gesture Customization**: Provides flexibility for users to define and customize hand gestures for door control actions.
- **Status Display**: Displays the current status of the door (open or closed) and detected hand gestures in the video feed.

## Requirements

- Python 3.x
- OpenCV
- cvzone (HandTrackingModule)
- NumPy

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/hand-gesture-door-control.git
    ```

2. Install the required dependencies:

    ```bash
    pip install opencv-python cvzone numpy
    ```

## Usage

1. Run the `gesture_door_control.py` script:

    ```bash
    python gesture_door_control.py
    ```

2. Ensure that your webcam is properly connected and positioned to capture hand gestures.

3. Perform the predefined hand gestures to control the virtual door.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/my-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push your changes to the branch (`git push origin feature/my-feature`).
5. Create a new Pull Request.
