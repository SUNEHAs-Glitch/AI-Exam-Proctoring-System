import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import winsound
import os

# =========================================
# INITIALIZATION
# =========================================

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# YOLO Model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0)

# Create screenshot folder
os.makedirs("screenshots", exist_ok=True)

# =========================================
# VARIABLES
# =========================================

warning_score = 0
last_beep = 0
last_screenshot = 0
last_warning_time = 0
prev_time = 0

# Eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# =========================================
# FUNCTIONS
# =========================================

def beep():
    winsound.Beep(1200, 200)

def get_head_direction(landmarks, w, h):

    nose = landmarks[1]
    left_face = landmarks[234]
    right_face = landmarks[454]

    nose_x = nose.x * w
    nose_y = nose.y * h

    left_x = left_face.x * w
    right_x = right_face.x * w

    face_center = (left_x + right_x) / 2

    horizontal_offset = nose_x - face_center

    if horizontal_offset < -25:
        return "LEFT"

    elif horizontal_offset > 25:
        return "RIGHT"

    elif nose_y < h * 0.38:
        return "UP"

    elif nose_y > h * 0.65:
        return "DOWN"

    else:
        return "FORWARD"

def eye_aspect_ratio(landmarks, eye_points, w, h):

    points = [
        (
            int(landmarks[i].x * w),
            int(landmarks[i].y * h)
        )
        for i in eye_points
    ]

    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    return (A + B) / (2.0 * C)

def add_warning(points=1):

    global warning_score
    global last_warning_time

    if time.time() - last_warning_time > 1:
        warning_score += points
        last_warning_time = time.time()

def save_screenshot(frame):

    timestamp = int(time.time())

    filename = f"screenshots/alert_{timestamp}.jpg"

    cv2.imwrite(filename, frame)

# =========================================
# MAIN LOOP
# =========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1000, 700))

    h, w, _ = frame.shape

    # =========================================
    # MONITORING ZONE
    # =========================================

    cv2.rectangle(
        frame,
        (320, 150),
        (680, 550),
        (80, 80, 80),
        2
    )

    cv2.putText(
        frame,
        "AI MONITORING ZONE",
        (360, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 150, 150),
        2
    )

    # =========================================
    # FPS
    # =========================================

    curr_time = time.time()

    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0

    prev_time = curr_time

    # =========================================
    # FACE + YOLO PROCESSING
    # =========================================

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)

    yolo_results = model(frame, verbose=False)

    status = "NORMAL"
    status_color = (0, 255, 0)

    direction = "FORWARD"

    # =========================================
    # FACE ANALYSIS
    # =========================================

    if face_results.multi_face_landmarks:

        for face_landmarks in face_results.multi_face_landmarks:

            # FACE MESH
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=1,
                    circle_radius=1
                )
            )

            # HEAD DIRECTION
            direction = get_head_direction(
                face_landmarks.landmark,
                w,
                h
            )

            # EYE ANALYSIS
            left_ear = eye_aspect_ratio(
                face_landmarks.landmark,
                LEFT_EYE,
                w,
                h
            )

            right_ear = eye_aspect_ratio(
                face_landmarks.landmark,
                RIGHT_EYE,
                w,
                h
            )

            avg_ear = (left_ear + right_ear) / 2

            # HEAD ALERTS
            if direction != "FORWARD":

                status = f"LOOKING {direction}"
                status_color = (0, 0, 255)

                add_warning(1)

            # EYE ALERTS
            if avg_ear < 0.20:

                status = "EYES CLOSED"
                status_color = (0, 0, 255)

                add_warning(2)

    else:

        status = "NO FACE DETECTED"
        status_color = (0, 0, 255)

        add_warning(3)

    # =========================================
    # PHONE DETECTION
    # =========================================

    phone_detected = False

    for r in yolo_results:

        for box in r.boxes:

            cls = int(box.cls[0])

            # cellphone class
            if cls == 67:

                phone_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    "PHONE DETECTED",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                add_warning(5)

    # =========================================
    # HIGH RISK ALERT
    # =========================================

    if warning_score > 15:

        status = "HIGH RISK DETECTED"
        status_color = (0, 0, 255)

        # RED OVERLAY
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (1000, 700),
            (0, 0, 255),
            -1
        )

        alpha = 0.15

        frame = cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0
        )

        # BEEP
        if time.time() - last_beep > 2:

            beep()

            last_beep = time.time()

        # SCREENSHOT
        if time.time() - last_screenshot > 5:

            save_screenshot(frame)

            last_screenshot = time.time()

    # =========================================
    # SCORE DECAY
    # =========================================

    if warning_score > 0:

        if time.time() - last_warning_time > 3:

            warning_score -= 1

    # =========================================
    # UI PANELS
    # =========================================

    # TOP PANEL
    cv2.rectangle(
        frame,
        (0, 0),
        (1000, 120),
        (20, 20, 20),
        -1
    )

    # BOTTOM PANEL
    cv2.rectangle(
        frame,
        (0, 660),
        (1000, 700),
        (20, 20, 20),
        -1
    )

    # TITLE
    cv2.putText(
        frame,
        "AI EXAM PROCTOR SYSTEM",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # STATUS
    cv2.putText(
        frame,
        f"STATUS: {status}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )

    # WARNING SCORE
    cv2.putText(
        frame,
        f"WARNING SCORE: {warning_score}",
        (500, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2
    )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (500, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # RISK LEVEL
    risk = "LOW"

    if warning_score > 10:
        risk = "MEDIUM"

    if warning_score > 20:
        risk = "HIGH"

    risk_color = (0, 255, 0)

    if risk == "MEDIUM":
        risk_color = (0, 165, 255)

    if risk == "HIGH":
        risk_color = (0, 0, 255)

    cv2.putText(
        frame,
        f"RISK LEVEL: {risk}",
        (760, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        risk_color,
        2
    )

    # DIRECTION
    cv2.putText(
        frame,
        f"DIRECTION: {direction}",
        (760, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # =========================================
    # LIVE STATUS DOTS
    # =========================================

    cv2.circle(frame, (930, 35), 8, (0, 255, 0), -1)

    cv2.putText(
        frame,
        "CAMERA",
        (845, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.circle(frame, (930, 65), 8, (0, 255, 0), -1)

    cv2.putText(
        frame,
        "AI ACTIVE",
        (835, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # =========================================
    # WARNING BAR
    # =========================================

    bar_width = min(warning_score * 10, 300)

    cv2.rectangle(
        frame,
        (30, 620),
        (330, 645),
        (50, 50, 50),
        -1
    )

    cv2.rectangle(
        frame,
        (30, 620),
        (30 + bar_width, 645),
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        "RISK METER",
        (30, 610),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # FOOTER
    cv2.putText(
        frame,
        "Press Q to Exit System",
        (30, 688),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )

    # =========================================
    # SHOW WINDOW
    # =========================================

    cv2.imshow(
        "AI Proctor System - ULTIMATE EDITION",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================
# CLEANUP
# =========================================

cap.release()
cv2.destroyAllWindows()