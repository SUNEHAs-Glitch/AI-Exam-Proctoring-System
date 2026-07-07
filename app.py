# Step 3 — FULL PROFESSIONAL STREAMLIT UI CODE


import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image
import os
from streamlit_javascript import st_javascript

# screenshot disabled for entire page

st_javascript("""
document.addEventListener("keyup", function(e) {

    if (e.key === "PrintScreen") {

        alert("Screenshots are not allowed!");

    }

});
""")
# right click disabled for entire page

st_javascript("""
document.addEventListener('contextmenu', event => event.preventDefault());
""")

# copy disabled for all elements

st_javascript("""
document.oncopy = function() {
    alert("Copy Disabled");
    return false;
}
""")

# =========================================
# SESSION STATE
# =========================================


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI Proctor System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI';
}

.main {
    background-color: #0e1117;
    color: white;
}

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #111827);
}

.metric-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #334155;
}

.alert-box {
    background-color: #7f1d1d;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-weight: bold;
}

.ok-box {
    background-color: #14532d;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# PROFESSIONAL LOGIN UI
# =========================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.markdown("""
    <style>

    .login-container{
        background: rgba(30,41,59,0.95);
        padding:40px;
        border-radius:20px;
        border:1px solid #334155;
        width:500px;
        margin:auto;
        margin-top:60px;
        box-shadow:0 0 30px rgba(0,0,0,0.4);
    }

    .login-title{
        text-align:center;
        color:white;
        font-size:38px;
        font-weight:bold;
        margin-bottom:10px;
    }

    .login-subtitle{
        text-align:center;
        color:#94a3b8;
        margin-bottom:30px;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-container">
        <div class="login-title">
            🛡️ AI EXAM PORTAL
        </div>

       
        Secure Autonomous Examination System
        
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        student_name = st.text_input(
            "Student Name",
            placeholder="Enter Full Name"
        )

        roll_no = st.text_input(
            "Roll Number",
            placeholder="Enter Roll Number"
        )

        subject = st.text_input(
            "Subject",
            placeholder="Enter Subject"
        )
        uploaded_image = st.file_uploader("Upload Your Face Image", type=["jpg", "png"])

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("START SECURE EXAM", use_container_width=True):

            if student_name and roll_no and subject:

                st.session_state.logged_in = True
                st.session_state.student_name = student_name
                st.session_state.roll_no = roll_no
                st.session_state.subject = subject

                st.rerun()

            else:
                st.error("Please Fill All Fields")

    st.stop()

 # =========================================
# LOGIN BUTTON + FACE SAVE
# =========================================

student_name = st.text_input("Student Name")

roll_no = st.text_input("Roll Number")

uploaded_image = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "png", "jpeg"]
)

if st.button("Start Exam"):

    if student_name and roll_no and uploaded_image:

        import os

        st.session_state.logged_in = True
        st.session_state.student_name = student_name
        st.session_state.roll_no = roll_no

        # create dataset folder
        os.makedirs("dataset", exist_ok=True)

        # save image
        image_path = f"dataset/{roll_no}.jpg"

        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.session_state.face_path = image_path

        st.rerun()

    else:
        st.error("Fill all fields + upload face image")

# =========================================
# TITLE
# =========================================

# =========================================
# PROFESSIONAL STUDENT SIDEBAR
# =========================================

st.sidebar.markdown("""
<div style="
background:#1e293b;
padding:20px;
border-radius:15px;
border:1px solid #334155;
text-align:center;
margin-bottom:20px;
">

<h2 style="color:white;">
🛡️ AI EXAM
</h2>

<p style="color:#94a3b8;">
Autonomous Proctor Agent
</p>

</div>
""", unsafe_allow_html=True)

if "student_name" in st.session_state:

    st.sidebar.markdown(f"""
    <div style="
    background:#0f172a;
    padding:18px;
    border-radius:15px;
    border:1px solid #334155;
    color:white;
    margin-bottom:20px;
    ">

    <h4>👨‍🎓 STUDENT INFO</h4>

    <hr>

    <p><b>Name:</b><br>{st.session_state.student_name}</p>

    <p><b>Roll No:</b><br>{st.session_state.roll_no}</p>

    <p><b>Subject:</b><br>{st.session_state.subject}</p>

    </div>
    """, unsafe_allow_html=True)

# =========================================
# SIDEBAR system controls
# =========================================

st.sidebar.markdown(f"""
    <div style="
    background:#0f172a;
    padding:18px;
    border-radius:15px;
    border:1px solid #334155;
    color:white;
    margin-bottom:20px;
    ">

    <h4>👨‍🎓 System Controls</h4>

    </div>
    """, unsafe_allow_html=True)

st.sidebar.title("System Controls")

camera_toggle = st.sidebar.toggle("Start Camera", value=True)

show_mesh = st.sidebar.checkbox("Face Mesh", value=True)
show_phone_detection = st.sidebar.checkbox("Phone Detection", value=True)
show_eye_tracking = st.sidebar.checkbox("Eye Tracking", value=True)
show_head_direction = st.sidebar.checkbox("Head Direction", value=True)
risk_threshold = st.sidebar.slider(
    "High Risk Threshold",
    10,
    100,
    20
)
tab_change = st_javascript(
    """
    await new Promise((resolve) => {

        document.addEventListener("visibilitychange", () => {

            if (document.hidden) {
                resolve("hidden");
            }
            else {
                resolve("visible");
            }

        });

    });
    """
)


# =========================================
# MODELS
# =========================================

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

model = YOLO("yolov8n.pt")

# =========================================
# VARIABLES
# =========================================

warning_score = 0

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# =========================================
# FUNCTIONS
# =========================================


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

# =========================================
# UI LAYOUT
# =========================================

col1, col2, col3, col4 = st.columns(4)

risk_placeholder = col1.empty()
status_placeholder = col2.empty()
face_placeholder = col3.empty()
fps_placeholder = col4.empty()

video_placeholder = st.empty()

log_box = st.empty()


# =========================================
# SCREEN SHARE SECTION
# =========================================

st.markdown("## 🖥️ Screen Monitoring")

screen_share = st.toggle("Enable Screen Sharing")

if screen_share:

    st.info("""
    Browser will ask permission to share screen.

    Please select:
    ✔ Entire Screen
    ✔ Your Exam Window

    AI monitoring will continue during exam.
    """)

    st.components.v1.html("""
    <script>

    async function startScreenShare() {

        try {

            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: true
            });

            console.log("Screen sharing started");

        }

        catch(err) {

            console.log("Permission denied");

        }
    }

    startScreenShare();

    </script>
    """, height=0)

# =========================================
# CAMERA
# =========================================

if camera_toggle:

    cap = cv2.VideoCapture(0)

  
    prev_time = 0

    while True:

        success, frame = cap.read()

        if not success:
            st.error("Camera Not Working")
            break
          
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1000, 700))

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(rgb)

        direction = "FORWARD"
        status = "NORMAL"
        status_color = (0, 255, 0)
        if tab_change == "hidden":
             warning_score += 10
             status = "TAB SWITCH DETECTED"

             cv2.putText(
                 frame,
                 "TAB SWITCH DETECTED",
                   (250, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1
                    (0, 0, 255),
                    3
                )

        # =========================================
        # FACE ANALYSIS
        # =========================================

        if face_results.multi_face_landmarks:         
            face_count = len(face_results.multi_face_landmarks)

            if face_count > 1:
                status = "MULTIPLE FACES"
                status_color = (0, 0, 255)
                warning_score += 5

                cv2.putText(
                    frame,
                    "MULTIPLE FACES DETECTED",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            for face_landmarks in face_results.multi_face_landmarks:

                if show_mesh:
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

                direction = get_head_direction(
                    face_landmarks.landmark,
                    w,
                    h
                )

                if direction != "FORWARD":
                    status = f"LOOKING {direction}"
                    status_color = (0, 0, 255)
                    warning_score += 1

                if show_eye_tracking:

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

                    if avg_ear < 0.20:
                        status = "EYES CLOSED"
                        status_color = (0, 0, 255)
                        warning_score += 2

        else:

            status = "NO FACE DETECTED"
            status_color = (0, 0, 255)
            warning_score += 3

  
         

        # =========================================
        # PHONE DETECTION
        # =========================================

        if show_phone_detection:

            yolo_results = model(frame, verbose=False)

            for r in yolo_results:

                for box in r.boxes:

                    cls = int(box.cls[0])

                    if cls == 67:

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
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        warning_score += 10
                        status = "PHONE DETECTED"

        # =========================================
        # RISK LEVEL
        # =========================================

        risk = "LOW"

        if warning_score > 10:
            risk = "MEDIUM"

        if warning_score > risk_threshold:
            risk = "HIGH"

        # =========================================
        # FPS
        # =========================================

        curr_time = time.time()

        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0

        prev_time = curr_time

        # =========================================
        # OVERLAY UI
        # =========================================

        cv2.rectangle(frame, (0, 0), (1000, 90), (15, 23, 42), -1)

        cv2.putText(
            frame,
            "AI PROCTOR ACTIVE",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"STATUS: {status}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        # =========================================
        # STREAMLIT METRICS
        # =========================================

        risk_placeholder.metric("Risk Level", risk)
        status_placeholder.metric("Direction", direction)
        face_placeholder.metric("Warning Score", warning_score)
        fps_placeholder.metric("FPS", int(fps))

        if risk == "HIGH":
            log_box.markdown(
                '<div class="alert-box">⚠ HIGH RISK DETECTED</div>',
                unsafe_allow_html=True
            )
        else:
            log_box.markdown(
                '<div class="ok-box">✔ SYSTEM NORMAL</div>',
                unsafe_allow_html=True
            )

        # =========================================
        # DISPLAY VIDEO
        # =========================================

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

    cap.release()

else:
    st.warning("Camera Disabled")