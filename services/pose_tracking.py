import cv2
import mediapipe as mp
import numpy as np
import base64
import asyncio

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Drawing landmarks

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return 360 - angle if angle > 180 else angle

def encode_frame(image):
    """Encode image as base64 to send over WebSocket."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

async def detect_pose():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Ensure compatibility on Windows
    if not cap.isOpened():
        print("Error: Camera not detected!")
        return

    counter, stage = 0, "down"
    exercise_started = False
    hand_raised_frames = 0  # Track frames where the hand is raised

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            pose_data = {"angle": None, "stage": stage, "counter": counter, "message": "Waiting for wave..."}

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                h, w, _ = image.shape

                # Get right hand positions (shoulder and wrist)
                right_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                right_shoulder_px = (int(right_shoulder[0] * w), int(right_shoulder[1] * h))
                right_wrist_px = (int(right_wrist[0] * w), int(right_wrist[1] * h))

                # Draw circles for right hand detection
                cv2.circle(image, right_shoulder_px, 10, (0, 0, 255), -1)  # Shoulder: Red
                cv2.circle(image, right_wrist_px, 10, (0, 255, 0), -1)  # Wrist: Green

                # Detect if the right hand is raised
                if right_wrist[1] < right_shoulder[1] - 0.05:  # Adjust threshold
                    hand_raised_frames += 1
                else:
                    hand_raised_frames = 0  # Reset if hand is not raised

                # Start exercise if the hand is raised for a few frames
                if hand_raised_frames > 5:  # Prevent flickering
                    exercise_started = True
                    pose_data["message"] = "Exercise started!"

                if exercise_started:
                    # Get left arm positions (shoulder, elbow, wrist)
                    shoulder, elbow, wrist = [
                        [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                        [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                        [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    ]

                    angle = calculate_angle(shoulder, elbow, wrist)
                    pose_data["angle"] = angle

                    # Exercise logic
                    if angle > 160 and stage == "up":
                        stage = "down"
                    elif angle < 60 and stage == "down":
                        stage = "up"
                        counter += 1

                    pose_data["stage"] = stage
                    pose_data["counter"] = counter

                    if angle > 160 or angle < 60:
                        pose_data["message"] = "Correct exercise"
                    else:
                        pose_data["message"] = "Wrong exercise!"

                    # Draw landmarks & text overlay
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.putText(image, f"Angle: {int(angle)}°", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(image, f"Stage: {stage}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.putText(image, f"Count: {counter}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Encode frame
                encoded_frame = encode_frame(image)
                pose_data["frame"] = encoded_frame

            yield pose_data
            await asyncio.sleep(0.05)

    cap.release()
