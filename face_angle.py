# import cv2
# import mediapipe as mp
# import numpy as np

# mp_face = mp.solutions.face_mesh
# face_mesh = mp_face.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# cap = cv2.VideoCapture(0)

# def get_head_direction(landmarks, w, h):
#     nose = landmarks[1]  # nose tip

#     x = int(nose.x * w)
#     y = int(nose.y * h)

#     if x < w * 0.4:
#         return "LEFT"
#     elif x > w * 0.6:
#         return "RIGHT"
#     elif y < h * 0.4:
#         return "UP"
#     elif y > h * 0.7:
#         return "DOWN"
#     else:
#         return "FORWARD"

# while True:
#     success, frame = cap.read()
#     if not success:
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)

#     status = "NO FACE"
#     color = (0, 0, 255)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:

#             direction = get_head_direction(face_landmarks.landmark, w, h)

#             if direction == "FORWARD":
#                 status = "OK - FACE NORMAL"
#                 color = (0, 255, 0)
#             else:
#                 status = f"WARNING: LOOKING {direction}"
#                 color = (0, 0, 255)

#     cv2.putText(frame, status, (30, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#     cv2.imshow("AI Exam Agent - Face Angle", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# stats
prev_time = 0
warning_score = 0
last_beep = 0


def get_head_direction(landmarks, w, h):
    nose = landmarks[1]

    x = int(nose.x * w)
    y = int(nose.y * h)

    if x < w * 0.4:
        return "LEFT"
    elif x > w * 0.6:
        return "RIGHT"
    elif y < h * 0.4:
        return "UP"
    elif y > h * 0.7:
        return "DOWN"
    else:
        return "FORWARD"


def beep():
    winsound.Beep(1000, 150)


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "NO FACE DETECTED"
    color = (0, 0, 255)
    direction = "NONE"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            direction = get_head_direction(face_landmarks.landmark, w, h)

            if direction == "FORWARD":
                status = "FACE NORMAL"
                color = (0, 255, 0)
            else:
                status = f"SUSPICIOUS: LOOKING {direction}"
                color = (0, 0, 255)
                warning_score += 1

                # beep cooldown
                if time.time() - last_beep > 2:
                    beep()
                    last_beep = time.time()

    else:
        warning_score += 2

    # ===== UI PANEL =====
     
    # cv2.rectangle(frame, (10, 10), (500, 140), (30, 30, 30), -1)
        cv2.putText(frame, "AI PROCTOR SYSTEM", (20, 35), ...)
        cv2.putText(frame, f"STATUS: {status}", (20, 60), ...)
        cv2.putText(frame, f"DIRECTION: {direction}", (20, 85), ...)
        cv2.putText(frame, f"WARNING: {warning_score}", (20, 105), ...)

    cv2.putText(frame, "AI PROCTOR SYSTEM", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"STATUS: {status}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"DIRECTION: {direction}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.putText(frame, f"WARNING SCORE: {warning_score}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (380, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # ===== FACE BOX =====
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            x = int(nose.x * w)
            y = int(nose.y * h)

            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.putText(frame, "FACE TRACKED", (x - 50, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("AI Exam Proctor - PROFESSIONAL MODE", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()