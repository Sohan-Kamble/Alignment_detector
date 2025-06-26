# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
from collections import deque
import math

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def classify_alignment_live(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    alignment_threshold = 5
    center_buffer = deque(maxlen=5)
    
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        frame_center = (width // 2, height // 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=80
        )

        cv2.line(frame, (frame_center[0], 0), (frame_center[0], height), (0, 255, 0), 1)
        cv2.line(frame, (0, frame_center[1]), (width, frame_center[1]), (0, 255, 0), 1)

        status = "NOT ALIGNED"
        color = (0, 0, 255)

        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            min_dist = float('inf')
            best_circle = None

            for (x, y, r) in circles:
                dist = euclidean_distance((x, y), frame_center)
                if dist < min_dist:
                    min_dist = dist
                    best_circle = (x, y, r)

            if best_circle:
                x, y, r = best_circle
                center_buffer.append((x, y))
                avg_x = int(np.mean([pt[0] for pt in center_buffer]))
                avg_y = int(np.mean([pt[1] for pt in center_buffer]))

                cv2.circle(frame, (avg_x, avg_y), r, (255, 0, 0), 2)
                cv2.circle(frame, (avg_x, avg_y), 2, (0, 255, 255), 3)

                dist_to_center = euclidean_distance((avg_x, avg_y), frame_center)
                if dist_to_center <= alignment_threshold:
                    status = "ALIGNED"
                    color = (0, 255, 0)
        else:
            cv2.putText(frame, "No Circle Detected", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

# === Streamlit UI ===
st.title("Video Alignment Classifier")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if video_file is not None:
    st.success("Video uploaded successfully. Processing...")
    classify_alignment_live(video_file)
