# 
import cv2
import mediapipe as mp
import time
import winsound

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)

last_beep = 0

def beep():
    winsound.Beep(1000, 150)  # frequency, duration

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    status = "NO FACE DETECTED"
    color = (0, 0, 255)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Confidence
            score = int(det.score[0] * 100)

            status = f"FACE OK | CONFIDENCE: {score}%"
            color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            cv2.putText(frame, status, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    else:
        cv2.putText(frame, status, (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Beep cooldown (avoid spam)
        if time.time() - last_beep > 2:
            beep()
            last_beep = time.time()

    # PROFESSIONAL UI PANEL
    cv2.rectangle(frame, (10, 10), (400, 90), (30, 30, 30), -1)

    cv2.putText(frame, "AI PROCTOR SYSTEM", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, status, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("AI Exam Proctor - PRO MODE", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()