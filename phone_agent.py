import cv2
from ultralytics import YOLO
import time
import winsound

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

last_beep = 0
warning_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (800, 600))

    results = model(frame)

    phone_detected = False

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # COCO class 67 = cellphone
            if cls == 67:

                phone_detected = True
                warning_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 3)

                cv2.putText(frame,
                            "MOBILE PHONE DETECTED",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2)

    # ALERT PANEL
    cv2.rectangle(frame, (10, 10), (420, 120), (30, 30, 30), -1)

    cv2.putText(frame, "AI OBJECT DETECTION AGENT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    if phone_detected:

        cv2.putText(frame,
                    "STATUS: CHEATING OBJECT DETECTED",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)

        # beep cooldown
        if time.time() - last_beep > 2:
            winsound.Beep(1200, 200)
            last_beep = time.time()

    else:

        cv2.putText(frame,
                    "STATUS: CLEAN",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    cv2.putText(frame,
                f"WARNING SCORE: {warning_count}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2)

    cv2.imshow("AI Proctor - YOLO Phone Agent", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()