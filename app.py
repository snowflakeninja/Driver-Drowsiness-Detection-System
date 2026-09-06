import time
from pathlib import Path

import cv2
import dlib
import pandas as pd
import streamlit as st
from scipy.spatial import distance as dist
from imutils import face_utils


EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 15
PREDICTOR_PATH = Path("shape_predictor_68_face_landmarks.dat")


def calculate_ear(eye):
    vertical_1 = dist.euclidean(eye[1], eye[5])
    vertical_2 = dist.euclidean(eye[2], eye[4])
    horizontal = dist.euclidean(eye[0], eye[3])

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def play_alarm_sound():
    sound_url = (
        "https://actions.google.com/sounds/v1/alarms/"
        "digital_watch_alarm_long.ogg"
    )

    st.components.v1.html(
        f"""
        <audio autoplay>
            <source src="{sound_url}" type="audio/ogg">
        </audio>
        """,
        height=0,
        width=0
    )


st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    layout="wide"
)

st.title("Driver Drowsiness Detection System")
st.write(
    "Real-time driver fatigue monitoring using facial landmarks "
    "and Eye Aspect Ratio (EAR)."
)


if not PREDICTOR_PATH.exists():
    st.error(
        "Facial landmark model not found. "
        "Place 'shape_predictor_68_face_landmarks.dat' "
        "in the same folder as app.py."
    )
    st.stop()


if "system_active" not in st.session_state:
    st.session_state.system_active = False


col1, col2 = st.columns(2)

with col1:
    if st.button(
        "Start System",
        type="primary",
        use_container_width=True
    ):
        st.session_state.system_active = True

with col2:
    if st.button(
        "Stop System",
        use_container_width=True
    ):
        st.session_state.system_active = False


st.divider()


left_col, right_col = st.columns([5, 3])


with left_col:
    st.subheader("Live Video Feed")
    video_placeholder = st.empty()


with right_col:
    st.subheader("System Metrics")

    status_placeholder = st.empty()

    metric1, metric2 = st.columns(2)

    with metric1:
        ear_placeholder = st.empty()

    with metric2:
        alert_count_placeholder = st.empty()

    st.subheader("Eye Aspect Ratio")
    chart_placeholder = st.empty()


alert_placeholder = st.empty()


detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    str(PREDICTOR_PATH)
)

left_start, left_end = (
    face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
)

right_start, right_end = (
    face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
)


frame_counter = 0
total_alerts = 0
ear_history = []


status_placeholder.info(
    "System Status: Idle\n\n"
    "Click Start System to begin monitoring."
)

ear_placeholder.metric(
    "Current EAR",
    "0.00"
)

alert_count_placeholder.metric(
    "Drowsy Alerts",
    "0"
)


if st.session_state.system_active:

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("Unable to access the webcam.")
        st.session_state.system_active = False
        st.stop()

    while camera.isOpened() and st.session_state.system_active:

        success, frame = camera.read()

        if not success:
            st.error("Unable to read video from the webcam.")
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = detector(gray, 0)

        current_ear = 0.0

        alert_placeholder.empty()


        if len(faces) == 0:

            frame_counter = 0

            status_placeholder.warning(
                "System Status: No Face Detected\n\n"
                "Please position yourself in front of the camera."
            )


        else:

            # Use the largest detected face as the driver
            face = max(
                faces,
                key=lambda rect: rect.width() * rect.height()
            )

            x = face.left()
            y = face.top()
            w = face.width()
            h = face.height()

            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)


            for point_x, point_y in shape:
                cv2.circle(
                    frame,
                    (point_x, point_y),
                    2,
                    (255, 255, 0),
                    -1
                )


            left_eye = shape[
                left_start:left_end
            ]

            right_eye = shape[
                right_start:right_end
            ]


            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            current_ear = (
                left_ear + right_ear
            ) / 2.0


            ear_history.append(current_ear)

            if len(ear_history) > 50:
                ear_history.pop(0)


            left_hull = cv2.convexHull(left_eye)
            right_hull = cv2.convexHull(right_eye)

            cv2.drawContours(
                frame,
                [left_hull],
                -1,
                (0, 255, 0),
                1
            )

            cv2.drawContours(
                frame,
                [right_hull],
                -1,
                (0, 255, 0),
                1
            )


            if current_ear < EAR_THRESHOLD:

                frame_counter += 1

                if frame_counter >= FRAME_THRESHOLD:

                    status_placeholder.error(
                        "System Status: Drowsiness Detected"
                    )

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT",
                        (x, max(y - 15, 30)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    if frame_counter == FRAME_THRESHOLD:

                        total_alerts += 1

                        with alert_placeholder:
                            play_alarm_sound()

                else:

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 165, 255),
                        2
                    )


            else:

                frame_counter = 0

                status_placeholder.success(
                    "System Status: Active\n\n"
                    "Driver is alert."
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )


        ear_placeholder.metric(
            "Current EAR",
            f"{current_ear:.2f}"
        )

        alert_count_placeholder.metric(
            "Drowsy Alerts",
            total_alerts
        )


        if ear_history:

            ear_df = pd.DataFrame(
                ear_history,
                columns=["Eye Aspect Ratio"]
            )

            chart_placeholder.line_chart(
                ear_df,
                height=180
            )


        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        video_placeholder.image(
            rgb_frame,
            channels="RGB",
            use_container_width=True
        )

        time.sleep(0.01)


    camera.release()