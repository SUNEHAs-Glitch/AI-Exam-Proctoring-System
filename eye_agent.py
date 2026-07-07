import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Eye landmarks (MediaPipe)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_points, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]

    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (A + B) / (2.0 * C)
    return ear

last_alert = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "NO FACE"
    color = (0, 0, 255)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face.landmark, RIGHT_EYE, w, h)

            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < 0.20:
                status = "EYES CLOSED - WARNING"
                color = (0, 0, 255)

                if time.time() - last_alert > 2:
                    winsound.Beep(1200, 200)
                    last_alert = time.time()
            else:
                status = "EYES OPEN - OK"
                color = (0, 255, 0)

    cv2.putText(frame, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Eye Tracking AI Agent", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()