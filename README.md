# Driver Drowsiness Detection System

A real-time driver drowsiness detection application built using Python, OpenCV, dlib, and Streamlit.

The system uses facial landmark detection and the Eye Aspect Ratio (EAR) to monitor eye closure. If the driver's eyes remain below a defined EAR threshold for a consecutive number of frames, the system identifies possible drowsiness and generates an alert.

## Features

- Real-time webcam monitoring
- Face detection using dlib
- 68-point facial landmark detection
- Eye tracking using facial landmarks
- Eye Aspect Ratio (EAR) calculation
- Consecutive-frame drowsiness detection
- Audio warning when drowsiness is detected
- Live EAR monitoring
- Drowsiness alert counter
- Real-time EAR trend chart
- Streamlit-based user interface

## Technologies Used

- Python
- Streamlit
- OpenCV
- dlib
- SciPy
- imutils
- Pandas

## Project Structure

```text
driver-drowsiness-detection/
│
├── app.py
├── requirements.txt
└── README.md
```

## How Drowsiness Detection Works

The system monitors the driver's eyes using facial landmarks detected from the webcam feed.

The main steps are:

1. Capture frames from the webcam.
2. Convert each frame to grayscale.
3. Detect the face using dlib's frontal face detector.
4. Detect 68 facial landmarks using the pretrained landmark predictor.
5. Extract the landmarks corresponding to the left and right eyes.
6. Calculate the Eye Aspect Ratio for each eye.
7. Calculate the average EAR of both eyes.
8. Compare the EAR with a predefined threshold.
9. Track how many consecutive frames remain below the threshold.
10. Generate an alert when the frame threshold is reached.

## Eye Aspect Ratio

Eye Aspect Ratio is used to estimate whether an eye is open or closed.

For six eye landmarks, EAR is calculated as:

```text
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

The vertical distances between the eye landmarks decrease when the eye closes, causing the EAR value to decrease.

The implementation used in the project is:

```python
def calculate_ear(eye):
    vertical_1 = dist.euclidean(eye[1], eye[5])
    vertical_2 = dist.euclidean(eye[2], eye[4])
    horizontal = dist.euclidean(eye[0], eye[3])

    return (vertical_1 + vertical_2) / (2.0 * horizontal)
```

## Drowsiness Detection Logic

The application uses the following configuration:

```python
EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 15
```

If the calculated EAR falls below `0.25`, the system begins counting consecutive frames.

If the EAR remains below the threshold for at least 15 consecutive frames, the system treats this as a possible drowsiness event and generates an alert.

If the EAR rises above the threshold, the consecutive-frame counter is reset.

This approach helps avoid triggering an alert for normal short blinks.

## Facial Landmark Detection

The project uses dlib's pretrained 68-point facial landmark predictor:

```text
shape_predictor_68_face_landmarks.dat
```

The landmarks corresponding to the left and right eyes are extracted and used for EAR calculation.

The application monitors the largest detected face in the frame, which is treated as the primary subject.

## Dashboard

The Streamlit interface displays:

- Live webcam feed
- Facial landmarks
- Eye contours
- Current Eye Aspect Ratio
- System status
- Number of drowsiness alerts
- Live EAR trend chart
- Visual warning when drowsiness is detected

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd driver-drowsiness-detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Requirements

The project uses the following Python packages:

```text
streamlit
pandas
opencv-python
dlib
scipy
imutils
```

## Running the Application

Make sure the facial landmark predictor file is present in the project directory:

```text
shape_predictor_68_face_landmarks.dat
```

Start the application:

```bash
streamlit run app.py
```

The application will usually be available at:

```text
http://localhost:8501
```

Click **Start System** to begin webcam monitoring.

Click **Stop System** to stop the monitoring process.

## Alert Logic

The system uses both an EAR threshold and a consecutive-frame threshold rather than generating an alert immediately when the eyes close.

For example:

```text
Eyes Open
   |
   v
EAR >= 0.25
   |
Driver Alert

Eyes Closing
   |
   v
EAR < 0.25
   |
Count Consecutive Frames
   |
   v
Frames >= 15
   |
Drowsiness Alert
```

This helps distinguish a normal blink from prolonged eye closure.

## Limitations

- EAR thresholds may need adjustment for different users and camera positions.
- Performance depends on lighting conditions and webcam quality.
- Facial landmarks may be less reliable when the face is significantly rotated or partially obstructed.
- The current system primarily detects prolonged eye closure and does not consider all possible indicators of driver fatigue.
- The application is intended as a computer vision demonstration and should not be used as a substitute for certified driver-monitoring or vehicle safety systems.
- Webcam access is designed primarily for local execution.

## Future Improvements

- Detect yawning using mouth landmarks
- Add head-pose estimation
- Make EAR thresholds adaptive for individual users
- Track blink frequency
- Record drowsiness events with timestamps
- Add monitoring-session statistics
- Improve detection under different lighting conditions
- Explore deep-learning-based fatigue detection
