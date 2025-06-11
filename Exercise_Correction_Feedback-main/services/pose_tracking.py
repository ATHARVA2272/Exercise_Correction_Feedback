# import threading
# import cv2
# import mediapipe as mp
# import numpy as np
# import base64
# import asyncio

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils  # Drawing landmarks

# def calculate_angle(a, b, c):
#     """Calculate the angle between three points."""
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(np.degrees(radians))
#     return 360 - angle if angle > 180 else angle

# def encode_frame(image):
#     """Encode image as base64 to send over WebSocket."""
#     _, buffer = cv2.imencode('.jpg', image)
#     return base64.b64encode(buffer).decode('utf-8')

# class VideoCaptureThread(threading.Thread):
#     def __init__(self):
#         super().__init__()
#         self.capture = cv2.VideoCapture(0)
#         self.frame = None
#         self.lock = threading.Lock()
#         self.running = True

#     def run(self):
#         while self.running:
#             ret, frame = self.capture.read()
#             if ret:
#                 with self.lock:
#                     self.frame = frame

#     def get_frame(self):
#         with self.lock:
#             return self.frame

#     def stop(self):
#         self.running = False
#         self.capture.release()

# # Initialize and start video capture in a separate thread
# video_thread = VideoCaptureThread()
# video_thread.start()

# async def detect_pose(exercise_type):
#     counter, stage = 0, "down"

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while True:
#             frame = video_thread.get_frame()
#             if frame is None:
#                 await asyncio.sleep(0.1)
#                 continue

#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(image)
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             pose_data = {"exercise": exercise_type, "counter": counter, "message": "Waiting for movement..."}

#             if results.pose_landmarks:
#                 lm = results.pose_landmarks.landmark
#                 h, w, _ = image.shape

#                 if exercise_type == "biceps":
#                     result = biceps_curl(lm, w, h, stage, counter)
#                 elif exercise_type == "squat":
#                     result = squat_counter(lm, w, h, stage, counter)
#                 elif exercise_type == "plank":
#                     result = plank_hold(lm, w, h)
#                 elif exercise_type == "lunges":
#                     result = lunge_counter(lm, w, h, stage, counter)
#                 else:
#                     result = {"message": "Invalid exercise type"}

#                 counter, stage = result.get("counter", counter), result.get("stage", stage)
#                 pose_data.update(result)

#                 # Draw pose landmarks
#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             pose_data["frame"] = encode_frame(image)
#             yield pose_data  # Send pose data to WebSocket
#             await asyncio.sleep(0.05)

# # Biceps Curl Detection
# def biceps_curl(lm, w, h, stage, counter):
#     """Detect biceps curl and count reps."""
#     shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#     elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#     wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#     angle = calculate_angle(shoulder, elbow, wrist)
    
#     if angle > 160 and stage == "up":
#         stage = "down"
#     elif angle < 60 and stage == "down":
#         stage = "up"
#         counter += 1

#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Correct movement" if 60 <= angle <= 160 else "Incorrect!"}

# # Squat Detection
# def squat_counter(lm, w, h, stage, counter):
#     """Detect squats and count reps."""
#     hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#     knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
#     ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

#     angle = calculate_angle(hip, knee, ankle)

#     if angle > 160 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 70 and stage == "up":
#         stage = "down"

#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Squat detected!"}

# # Plank Hold Detection
# def plank_hold(lm, w, h):
#     """Detect correct plank form."""
#     shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#     hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#     knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

#     angle = calculate_angle(shoulder, hip, knee)

#     return {"angle": angle, "message": "Good plank!" if 160 <= angle <= 180 else "Adjust position!"}

# # Lunge Detection
# def lunge_counter(lm, w, h, stage, counter):
#     """Detect lunges and count reps."""
#     hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#     knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
#     ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

#     angle = calculate_angle(hip, knee, ankle)

#     if angle > 160 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 70 and stage == "up":
#         stage = "down"

#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Lunge detected!"}

# # Shutdown video capture when the server stops
# import signal
# import sys

# def shutdown_handler(signal, frame):
#     print("Shutting down...")
#     video_thread.stop()
#     sys.exit(0)

# signal.signal(signal.SIGINT, shutdown_handler)


# import threading
# import cv2
# import mediapipe as mp
# import numpy as np
# import base64
# import asyncio
# import websockets

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(np.degrees(radians))
#     return 360 - angle if angle > 180 else angle

# def encode_frame(image):
#     _, buffer = cv2.imencode('.jpg', image)
#     return base64.b64encode(buffer).decode('utf-8')

# class VideoCaptureThread(threading.Thread):
#     def __init__(self):
#         super().__init__()
#         self.capture = cv2.VideoCapture(0)
#         self.frame = None
#         self.lock = threading.Lock()
#         self.running = True

#     def run(self):
#         while self.running:
#             ret, frame = self.capture.read()
#             if ret:
#                 with self.lock:
#                     self.frame = frame

#     def get_frame(self):
#         with self.lock:
#             return self.frame

#     def stop(self):
#         self.running = False
#         self.capture.release()

# video_thread = VideoCaptureThread()
# video_thread.start()

# async def detect_pose(websocket, path):
#     exercise_type = path.split('/')[-1]
#     counter, stage = 0, "down"

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while True:
#             frame = video_thread.get_frame()
#             if frame is None:
#                 await asyncio.sleep(0.1)
#                 continue

#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(image)
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             pose_data = {"exercise": exercise_type, "counter": counter, "message": "Waiting for movement..."}

#             if results.pose_landmarks:
#                 lm = results.pose_landmarks.landmark
#                 h, w, _ = image.shape

#                 if exercise_type == "biceps":
#                     result = biceps_curl(lm, stage, counter)
#                 elif exercise_type == "squat":
#                     result = squat_counter(lm, stage, counter)
#                 elif exercise_type == "plank":
#                     result = plank_hold(lm)
#                 elif exercise_type == "lunges":
#                     result = lunge_counter(lm, stage, counter)
#                 else:
#                     result = {"message": "Invalid exercise type"}

#                 counter, stage = result.get("counter", counter), result.get("stage", stage)
#                 pose_data.update(result)
#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             pose_data["frame"] = encode_frame(image)
#             await websocket.send(base64.b64encode(str(pose_data).encode()).decode())
#             await asyncio.sleep(0.05)

# def biceps_curl(lm, stage, counter):
#     shoulder, elbow, wrist = lm[11], lm[13], lm[15]
#     angle = calculate_angle((shoulder.x, shoulder.y), (elbow.x, elbow.y), (wrist.x, wrist.y))
#     if angle > 160 and stage == "up":
#         stage = "down"
#     elif angle < 60 and stage == "down":
#         stage = "up"
#         counter += 1
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Good form" if 60 <= angle <= 160 else "Adjust movement"}

# def squat_counter(lm, stage, counter):
#     hip, knee, ankle = lm[23], lm[25], lm[27]
#     angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
#     if angle > 160 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 70 and stage == "up":
#         stage = "down"
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Keep squatting!"}

# def plank_hold(lm):
#     shoulder, hip, knee = lm[11], lm[23], lm[25]
#     angle = calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
#     return {"angle": angle, "message": "Hold steady!" if 160 <= angle <= 180 else "Fix posture!"}

# def lunge_counter(lm, stage, counter):
#     hip, knee, ankle = lm[23], lm[25], lm[27]
#     angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
#     if angle > 160 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 70 and stage == "up":
#         stage = "down"
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Lunge detected!"}

# import signal
# import sys

# def shutdown_handler(signal, frame):
#     print("Shutting down...")
#     video_thread.stop()
#     sys.exit(0)

# signal.signal(signal.SIGINT, shutdown_handler)

# start_server = websockets.serve(detect_pose, "localhost", 8000)
# asyncio.create_task(start_server())  # Call the function with ()
# asyncio.get_event_loop().run_forever()
# import threading
# import cv2
# import mediapipe as mp
# import numpy as np
# import base64
# import asyncio
# import websockets
# import signal
# import sys

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(np.degrees(radians))
#     return 360 - angle if angle > 180 else angle

# def encode_frame(image):
#     _, buffer = cv2.imencode('.jpg', image)
#     return base64.b64encode(buffer).decode('utf-8')

# class VideoCaptureThread(threading.Thread):
#     def __init__(self):
#         super().__init__()
#         self.capture = cv2.VideoCapture(0)
#         self.frame = None
#         self.lock = threading.Lock()
#         self.running = True

#     def run(self):
#         while self.running:
#             ret, frame = self.capture.read()
#             if ret:
#                 with self.lock:
#                     self.frame = frame

#     def get_frame(self):
#         with self.lock:
#             return self.frame

#     def stop(self):
#         self.running = False
#         self.capture.release()

# video_thread = VideoCaptureThread()
# video_thread.start()

# async def detect_pose(exercise_type):
#     counter, stage = 0, "down"

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while True:
#             frame = video_thread.get_frame()
#             if frame is None:
#                 await asyncio.sleep(0.1)
#                 continue

#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(image)
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             pose_data = {"exercise": exercise_type, "counter": counter, "message": "Waiting for movement..."}

#             if results.pose_landmarks:
#                 lm = results.pose_landmarks.landmark
                
#                 if exercise_type == "bicepCurl":
#                     result = biceps_curl(lm, stage, counter)
#                 elif exercise_type == "squat":
#                     result = squat_counter(lm, stage, counter)
#                 elif exercise_type == "plank":
#                     result = plank_hold(lm)
#                 elif exercise_type == "lunges":
#                     result = lunge_counter(lm, stage, counter)
#                 else:
#                     result = {"message": "Invalid exercise type"}

#                 counter, stage = result.get("counter", counter), result.get("stage", stage)
#                 pose_data.update(result)
#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             pose_data["frame"] = encode_frame(image)
#             yield pose_data
#             await asyncio.sleep(0.05)

# def biceps_curl(lm, stage, counter):
#     shoulder, elbow, wrist = lm[11], lm[13], lm[15]
#     angle = calculate_angle((shoulder.x, shoulder.y), (elbow.x, elbow.y), (wrist.x, wrist.y))
#     if angle > 160 and stage == "up":
#         stage = "down"
#     elif angle < 60 and stage == "down":
#         stage = "up"
#         counter += 1
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Good form" if 60 <= angle <= 160 else "Adjust movement"}

# def squat_counter(lm, stage, counter):
#     hip, knee, ankle = lm[23], lm[25], lm[27]
#     angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
#     if angle > 160 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 70 and stage == "up":
#         stage = "down"
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Keep squatting!"}

# def plank_hold(lm):
#     shoulder, hip, knee = lm[11], lm[23], lm[25]
#     angle = calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
#     return {"angle": angle, "message": "Hold steady!" if 160 <= angle <= 180 else "Fix posture!"}

# def lunge_counter(lm, stage, counter):
#     hip, knee, ankle = lm[23], lm[25], lm[27]
#     angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
#     if angle > 160 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 70 and stage == "up":
#         stage = "down"
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Lunge detected!"}

# def shutdown_handler(signal, frame):
#     print("Shutting down...")
#     video_thread.stop()
#     sys.exit(0)

# signal.signal(signal.SIGINT, shutdown_handler)

# async def main():
#     async with websockets.serve(detect_pose, "localhost", 8000):
#         await asyncio.Future()  # Keeps the server running indefinitely

# if __name__ == "__main__":
#     asyncio.run(main())




##########################################################################################################
##Final code
###########################################################################################################

# import cv2
# import mediapipe as mp
# import numpy as np
# import asyncio
# import base64
# import time
# import threading

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils


# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(np.degrees(radians))
#     return 360 - angle if angle > 180 else angle


# def encode_frame(image):
#     _, buffer = cv2.imencode('.jpg', image)
#     return base64.b64encode(buffer).decode('utf-8')


# class VideoCaptureThread(threading.Thread):
#     def __init__(self):
#         super().__init__()
#         self.capture = cv2.VideoCapture(0)
#         self.frame = None
#         self.lock = threading.Lock()
#         self.running = True

#     def run(self):
#         while self.running:
#             ret, frame = self.capture.read()
#             if ret:
#                 with self.lock:
#                     self.frame = frame

#     def get_frame(self):
#         with self.lock:
#             return self.frame

#     def stop(self):
#         self.running = False
#         self.capture.release()


# video_thread = VideoCaptureThread()
# video_thread.start()


# async def detect_pose(exercise_type):
#     """ Async generator that yields pose data for WebSocket streaming. """
#     counter, stage = 0, "down"
#     fps_time = time.time()

#     with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
#         while True:
#             frame = video_thread.get_frame()
#             if frame is None:
#                 await asyncio.sleep(0.01)
#                 continue

#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(image)
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             pose_data = {"counter": counter, "message": "Waiting for movement..."}

#             if results.pose_landmarks:
#                 lm = results.pose_landmarks.landmark

#                 if exercise_type == "bicepCurl":
#                     result = biceps_curl(lm, stage, counter)
#                 elif exercise_type == "squat":
#                     result = squat_counter(lm, stage, counter)
#                 elif exercise_type == "plank":
#                     result = plank_hold(lm)
#                 elif exercise_type == "lunges":
#                     result = lunge_counter(lm, stage, counter)
#                 else:
#                     result = {"message": "Invalid exercise type"}

#                 counter, stage = result.get("counter", counter), result.get("stage", stage)
#                 pose_data.update(result)
#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             pose_data["frame"] = encode_frame(image)
#             pose_data["fps"] = round(1.0 / (time.time() - fps_time), 2)
#             fps_time = time.time()

#             yield pose_data  # ✅ Yield instead of sending directly
#             await asyncio.sleep(0.03)  # Adjust for real-time responsiveness


# def biceps_curl(lm, stage, counter):
#     shoulder, elbow, wrist = lm[11], lm[13], lm[15]
#     angle = calculate_angle((shoulder.x, shoulder.y), (elbow.x, elbow.y), (wrist.x, wrist.y))
#     if angle > 165 and stage == "up":
#         stage = "down"
#     elif angle < 50 and stage == "down":
#         stage = "up"
#         counter += 1
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Good form"}


# def squat_counter(lm, stage, counter):
#     hip, knee, ankle = lm[23], lm[25], lm[27]
#     angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
#     if angle > 170 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 75 and stage == "up":
#         stage = "down"
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Keep squatting!"}


# def plank_hold(lm):
#     shoulder, hip, knee = lm[11], lm[23], lm[25]
#     angle = calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
#     return {"angle": angle, "message": "Hold steady!" if 165 <= angle <= 180 else "Fix posture!"}


# def lunge_counter(lm, stage, counter):
#     hip, knee, ankle = lm[23], lm[25], lm[27]
#     angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
#     if angle > 165 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 75 and stage == "up":
#         stage = "down"
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Lunge detected!"}


#Final 2.0
###########################################################################################################

# import cv2
# import mediapipe as mp
# import numpy as np
# import asyncio
# import base64
# import time
# import threading

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils


# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(np.degrees(radians))
#     return 360 - angle if angle > 180 else angle


# def encode_frame(image):
#     _, buffer = cv2.imencode('.jpg', image)
#     return base64.b64encode(buffer).decode('utf-8')


# class VideoCaptureThread(threading.Thread):
#     def __init__(self):
#         super().__init__()
#         self.capture = cv2.VideoCapture(0)
#         self.frame = None
#         self.lock = threading.Lock()
#         self.running = True

#     def run(self):
#         while self.running:
#             ret, frame = self.capture.read()
#             if ret:
#                 with self.lock:
#                     self.frame = frame

#     def get_frame(self):
#         with self.lock:
#             return self.frame

#     def stop(self):
#         self.running = False
#         self.capture.release()


# video_thread = VideoCaptureThread()
# video_thread.start()


# async def detect_pose(exercise_type):
#     """ Async generator that yields pose data for WebSocket streaming. """
#     counter, stage = 0, "down"
#     fps_time = time.time()

#     with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
#         while True:
#             frame = video_thread.get_frame()
#             if frame is None:
#                 await asyncio.sleep(0.01)
#                 continue

#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(image)
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             pose_data = {"counter": counter, "message": "Waiting for movement..."}

#             if results.pose_landmarks:
#                 lm = results.pose_landmarks.landmark

#                 if exercise_type == "bicepCurl":
#                     result = biceps_curl(lm, stage, counter)
#                 elif exercise_type == "squat":
#                     result = squat_counter(lm, stage, counter)
#                 elif exercise_type == "plank":
#                     result = plank_hold(lm)
#                 elif exercise_type == "lunges":
#                     result = lunge_counter(lm, stage, counter)
#                 else:
#                     result = {"message": "Invalid exercise type"}

#                 counter, stage = result.get("counter", counter), result.get("stage", stage)
#                 pose_data.update(result)
#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             pose_data["frame"] = encode_frame(image)
#             pose_data["fps"] = round(1.0 / (time.time() - fps_time), 2)
#             fps_time = time.time()

#             yield pose_data  # ✅ Yield instead of sending directly
#             await asyncio.sleep(0.03)  # Adjust for real-time responsiveness


# def biceps_curl(lm, stage, counter):
#     shoulder, elbow, wrist = lm[11], lm[13], lm[15]
#     angle = calculate_angle((shoulder.x, shoulder.y), (elbow.x, elbow.y), (wrist.x, wrist.y))
#     if angle > 165 and stage == "up":
#         stage = "down"
#     elif angle < 50 and stage == "down":
#         stage = "up"
#         counter += 1
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Good form"}


# def squat_counter(lm, stage, counter):
#     hip, knee, ankle = lm[23], lm[25], lm[27]
#     angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
#     if angle > 170 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 75 and stage == "up":
#         stage = "down"
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Keep squatting!"}


# def plank_hold(lm):
#     shoulder, hip, knee = lm[11], lm[23], lm[25]
#     angle = calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
#     return {"angle": angle, "message": "Hold steady!" if 165 <= angle <= 180 else "Fix posture!"}


# def lunge_counter(lm, stage, counter):
#     hip, knee, ankle = lm[23], lm[25], lm[27]
#     angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
#     if angle > 165 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 75 and stage == "up":
#         stage = "down"
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Lunge detected!"}


#################################################################################################################################################






# import cv2
# import mediapipe as mp
# import numpy as np
# import asyncio
# import base64
# import time
# import threading

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(np.degrees(radians))
#     return 360 - angle if angle > 180 else angle

# def encode_frame(image):
#     _, buffer = cv2.imencode('.jpg', image)
#     return base64.b64encode(buffer).decode('utf-8')

# class VideoCaptureThread(threading.Thread):
#     def __init__(self):
#         super().__init__()
#         self.capture = cv2.VideoCapture(0)
#         # Set camera resolution to capture full body
#         self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         self.frame = None
#         self.lock = threading.Lock()
#         self.running = True

#     def run(self):
#         while self.running:
#             ret, frame = self.capture.read()
#             if ret:
#                 with self.lock:
#                     self.frame = frame

#     def get_frame(self):
#         with self.lock:
#             return self.frame

#     def stop(self):
#         self.running = False
#         self.capture.release()


# video_thread = VideoCaptureThread()
# video_thread.start()

# async def detect_pose(exercise_type):
#     """ Async generator that yields pose data for WebSocket streaming. """
#     counter, stage = 0, "down"
#     fps_time = time.time()

#     with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
#         while True:
#             frame = video_thread.get_frame()
#             if frame is None:
#                 await asyncio.sleep(0.01)
#                 continue

#             # Reduce frame size for faster processing (optional)
#             small_frame = cv2.resize(frame, (640, 360))  # Reduced size for better performance
#             image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(image)
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             pose_data = {"counter": counter, "message": "Waiting for movement..."}

#             if results.pose_landmarks:
#                 lm = results.pose_landmarks.landmark

#                 if exercise_type == "bicepCurl":
#                     result = biceps_curl(lm, stage, counter)
#                 elif exercise_type == "squat":
#                     result = squat_counter(lm, stage, counter)
#                 elif exercise_type == "plank":
#                     result = plank_hold(lm)
#                 elif exercise_type == "lunges":
#                     result = lunge_counter(lm, stage, counter)
#                 else:
#                     result = {"message": "Invalid exercise type"}

#                 counter, stage = result.get("counter", counter), result.get("stage", stage)
#                 pose_data.update(result)
#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             pose_data["frame"] = encode_frame(image)
#             pose_data["fps"] = round(1.0 / (time.time() - fps_time), 2)
#             fps_time = time.time()

#             yield pose_data  # ✅ Yield instead of sending directly
#             await asyncio.sleep(0.01)  # Adjust for real-time responsiveness

# def biceps_curl(lm, stage, counter):
#     shoulder, elbow, wrist = lm[11], lm[13], lm[15]
#     angle = calculate_angle((shoulder.x, shoulder.y), (elbow.x, elbow.y), (wrist.x, wrist.y))
#     if angle > 165 and stage == "up":
#         stage = "down"
#     elif angle < 50 and stage == "down":
#         stage = "up"
#         counter += 1
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Good form"}

# def squat_counter(lm, stage, counter):
#     hip, knee, ankle = lm[23], lm[25], lm[27]
#     angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
#     if angle > 170 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 75 and stage == "up":
#         stage = "down"
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Keep squatting!"}

# def plank_hold(lm):
#     shoulder, hip, knee = lm[11], lm[23], lm[25]
#     angle = calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
#     return {"angle": angle, "message": "Hold steady!" if 165 <= angle <= 180 else "Fix posture!"}

# def lunge_counter(lm, stage, counter):
#     hip, knee, ankle = lm[23], lm[25], lm[27]
#     angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
#     if angle > 165 and stage == "down":
#         stage = "up"
#         counter += 1
#     elif angle < 75 and stage == "up":
#         stage = "down"
#     return {"angle": angle, "stage": stage, "counter": counter, "message": "Lunge detected!"}


import cv2
import mediapipe as mp
import numpy as np
import asyncio
import base64
import time
import threading

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_angle(a, b, c):
    """Calculate the angle between three points in 2D or 3D space."""
    if len(a) == 3 and len(b) == 3 and len(c) == 3:  # 3D points
        # Convert to numpy arrays for vector calculations
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Create vectors from points
        ba = a - b
        bc = c - b
        
        # Calculate cosine of angle using dot product
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine = np.clip(cosine, -1.0, 1.0)  # Ensure value is in valid range
        angle = np.degrees(np.arccos(cosine))
    else:  # 2D points
        a = np.array(a[:2])
        b = np.array(b[:2])
        c = np.array(c[:2])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(np.degrees(radians))
        if angle > 180:
            angle = 360 - angle
    
    return angle

def encode_frame(image):
    """Encode an image frame as base64 for WebSocket transmission."""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')

def visualize_angle(image, a, b, c, angle, color=(255, 255, 255)):
    """Draw the angle on the image with reference points and value."""
    cv2.line(image, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 2)
    cv2.line(image, (int(b[0]), int(b[1])), (int(c[0]), int(c[1])), color, 2)
    cv2.circle(image, (int(b[0]), int(b[1])), 5, color, -1)
    
    # Position text at the midpoint between b and center of image
    text_position = (int(b[0]), int(b[1]) - 20)
    cv2.putText(image, f"{angle:.1f}", text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return image

class VideoCaptureThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)
        # Set camera resolution to capture full body
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.capture.release()


video_thread = VideoCaptureThread()
video_thread.start()

def landmark_to_point(landmark, image_width, image_height):
    """Convert a normalized landmark to pixel coordinates with z-depth."""
    return (
        int(landmark.x * image_width),
        int(landmark.y * image_height),
        landmark.z
    )

def is_visible(landmark, visibility_threshold=0.5):
    """Check if a landmark is visible enough to be considered reliable."""
    return landmark.visibility > visibility_threshold



def lunge_counter(image, lm, width, height, stage, counter, good_form_streak):
    """
    Track lunges exercise with form correction feedback.
    """
    # Extract landmarks for both legs
    # Left leg
    left_hip = (lm[23].x * width, lm[23].y * height, lm[23].z)
    left_knee = (lm[25].x * width, lm[25].y * height, lm[25].z)
    left_ankle = (lm[27].x * width, lm[27].y * height, lm[27].z)
    
    # Right leg
    right_hip = (lm[24].x * width, lm[24].y * height, lm[24].z)
    right_knee = (lm[26].x * width, lm[26].y * height, lm[26].z)
    right_ankle = (lm[28].x * width, lm[28].y * height, lm[28].z)
    
    # Calculate angles
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    # Visualize angles
    visualize_angle(image, 
                   (left_hip[0], left_hip[1]), 
                   (left_knee[0], left_knee[1]), 
                   (left_ankle[0], left_ankle[1]), 
                   left_knee_angle, 
                   color=(255, 0, 0))
    
    visualize_angle(image, 
                   (right_hip[0], right_hip[1]), 
                   (right_knee[0], right_knee[1]), 
                   (right_ankle[0], right_ankle[1]), 
                   right_knee_angle, 
                   color=(0, 0, 255))
    
    # Check if landmarks are visible
    knee_visible = all(lm[i].visibility > 0.7 for i in [25, 26, 27, 28])
    
    form_feedback = ""
    message = ""
    
    if knee_visible:
        # Detect lunge stages based on knee angles
        # Track lunges - both front knee bend and back knee lowered is a rep
        if (left_knee_angle < 120 or right_knee_angle < 120) and stage == "up":
            stage = "down"
            message = "Lunge position reached"
        elif (left_knee_angle > 160 and right_knee_angle > 160) and stage == "down":
            stage = "up"
            counter += 1
            message = f"Lunge complete: {counter}"
            good_form_streak += 1
        
        # Form feedback
        if stage == "down":
            # Check front knee alignment (shouldn't go past toes)
            if left_knee_angle < right_knee_angle:  # Left leg is forward
                front_knee = left_knee
                front_ankle = left_ankle
            else:  # Right leg is forward
                front_knee = right_knee
                front_ankle = right_ankle
            
            # Check if front knee goes too far beyond toes
            if abs(front_knee[0] - front_ankle[0]) > 0.1 * width:
                form_feedback = "Front knee shouldn't go past toes"
                good_form_streak = max(0, good_form_streak - 1)
            else:
                form_feedback = "Good form, hold the position"
    else:
        form_feedback = "Position your legs in view of the camera"
    
    # Recognize good form streak
    if good_form_streak >= 3:
        form_feedback = "Excellent lunge form! Keep it up!"
    
    return {
        "left_knee_angle": round(left_knee_angle, 2),
        "right_knee_angle": round(right_knee_angle, 2),
        "stage": stage,
        "counter": counter,
        "message": message,
        "form_feedback": form_feedback,
        "good_form_streak": good_form_streak
    }

async def detect_pose(exercise_type):
    """ Async generator that yields pose data for WebSocket streaming. """
    # Initialize counters and states
    left_counter, right_counter = 0, 0 
    left_stage, right_stage = "down", "down"
    counter = 0  # Combined counter
    stage = "up" if exercise_type == "squat" else "down"  # Different initial states based on exercise
    
    # For time-based exercises
    start_time = None
    duration = 0
    
    # For form correction
    form_feedback = ""
    good_form_streak = 0
    
    # Performance metrics
    fps_time = time.time()
    
    # Configure pose detection with appropriate settings
    pose_config = mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1  # Using medium complexity for balance of speed and accuracy
    )
    
    with pose_config as pose:
        while True:
            frame = video_thread.get_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            # Process frame for pose detection
            h, w, _ = frame.shape
            # Optimize processing speed while maintaining accuracy
            process_frame = cv2.resize(frame, (int(w * 0.75), int(h * 0.75)))
            process_frame_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            process_frame_rgb.flags.writeable = False  # Performance optimization
            
            # Run pose detection
            results = pose.process(process_frame_rgb)
            
            # Prepare visualization frame
            process_frame_rgb.flags.writeable = True
            image = cv2.cvtColor(process_frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Default response if no landmarks detected
            pose_data = {
                "counter": counter,
                "left_counter": left_counter,
                "right_counter": right_counter,
                "message": "Waiting for movement...",
                "form_feedback": "Position yourself in the camera view"
            }

            if results.pose_landmarks:
                # Get landmarks and scale to original frame size
                lm = results.pose_landmarks.landmark
                scaled_lm = []
                for landmark in lm:
                    scaled_lm.append((
                        int(landmark.x * w),
                        int(landmark.y * h),
                        landmark.z,
                        landmark.visibility
                    ))
                
                # Process based on exercise type
                if exercise_type == "bicepCurl":
                    result = biceps_curl(image, lm, w, h, left_stage, right_stage, left_counter, right_counter, good_form_streak)
                    left_counter, right_counter = result.get("left_counter", left_counter), result.get("right_counter", right_counter)
                    left_stage, right_stage = result.get("left_stage", left_stage), result.get("right_stage", right_stage)
                    counter = left_counter + right_counter
                    good_form_streak = result.get("good_form_streak", good_form_streak)
                
                elif exercise_type == "squat":
                    result = squat_counter(image, lm, w, h, stage, counter, good_form_streak)
                    counter, stage = result.get("counter", counter), result.get("stage", stage)
                    good_form_streak = result.get("good_form_streak", good_form_streak)
                
                elif exercise_type == "plank":
                    if start_time is None:
                        start_time = time.time()
                    current_time = time.time()
                    duration = current_time - start_time
                    result = plank_hold(image, lm, w, h, duration, good_form_streak)
                    # Around line 1041 in detect_pose
                    if result is not None:
                        good_form_streak = result.get("good_form_streak", good_form_streak)
                    if result is not None:
                        result["duration"] = round(duration, 1)
                    else:
                        result = {"duration": round(duration, 1)}
                
                elif exercise_type == "lunges":
                    result = lunge_counter(image, lm, w, h, stage, counter, good_form_streak)
                    counter, stage = result.get("counter", counter), result.get("stage", stage)
                    good_form_streak = result.get("good_form_streak", good_form_streak)
                
                else:
                    result = {"message": "Invalid exercise type"}
                
                pose_data.update(result)
                pose_data["counter"] = counter
                
                # Draw the pose landmarks with custom style
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Resize back to display resolution
            display_frame = cv2.resize(image, (w, h))
            
            # Add exercise info overlay
            cv2.rectangle(display_frame, (0, 0), (300, 100), (0, 0, 0, 0.7), -1)
            cv2.putText(display_frame, f"Exercise: {exercise_type}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            if exercise_type == "bicepCurl":
                cv2.putText(display_frame, f"Left: {left_counter} | Right: {right_counter}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            elif exercise_type == "plank":
                cv2.putText(display_frame, f"Duration: {round(duration, 1)}s", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, f"Count: {counter}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add form feedback
            if "form_feedback" in pose_data and pose_data["form_feedback"]:
                # Create background for text
                text = pose_data["form_feedback"]
                (text_width, text_height), _ = cv2.getTextSize(
    text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)
                cv2.rectangle(display_frame, (10, h - 60), (10 + text_width, h - 20), 
                              (0, 0, 0, 0.7), -1)
                cv2.putText(display_frame, text, (10, h - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Calculate and display FPS
            current_time = time.time()
            fps = 1.0 / (current_time - fps_time)
            fps_time = current_time
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (w - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode frame and add to pose data
            pose_data["frame"] = encode_frame(display_frame)
            pose_data["fps"] = round(fps, 2)

            yield pose_data
            await asyncio.sleep(0.01)  # Yield control to allow other tasks to run

def biceps_curl(image, lm, width, height, left_stage, right_stage, left_counter, right_counter, good_form_streak):
    """
    Track biceps curl exercise for both arms with form correction feedback.
    """
    feedback = ""
    # Extract landmarks for both arms
    # Left arm (right side from camera view)
    left_shoulder = (lm[11].x * width, lm[11].y * height, lm[11].z)
    left_elbow = (lm[13].x * width, lm[13].y * height, lm[13].z)
    left_wrist = (lm[15].x * width, lm[15].y * height, lm[15].z)
    
    # Right arm (left side from camera view)
    right_shoulder = (lm[12].x * width, lm[12].y * height, lm[12].z)
    right_elbow = (lm[14].x * width, lm[14].y * height, lm[14].z)
    right_wrist = (lm[16].x * width, lm[16].y * height, lm[16].z)
    
    # Calculate angles
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Visualize angles on the image
    visualize_angle(image, 
                   (left_shoulder[0], left_shoulder[1]), 
                   (left_elbow[0], left_elbow[1]), 
                   (left_wrist[0], left_wrist[1]), 
                   left_angle, 
                   color=(255, 0, 0))
    
    visualize_angle(image, 
                   (right_shoulder[0], right_shoulder[1]), 
                   (right_elbow[0], right_elbow[1]), 
                   (right_wrist[0], right_wrist[1]), 
                   right_angle, 
                   color=(0, 0, 255))
    
    # Check visibility for reliable detection
    left_visible = all(lm[i].visibility > 0.7 for i in [11, 13, 15])
    right_visible = all(lm[i].visibility > 0.7 for i in [12, 14, 16])
    
    # Process left arm curl
    if left_visible:
        # Check for proper form
        left_hip = (lm[23].x * width, lm[23].y * height)
        left_shoulder_hip_x_diff = abs(left_shoulder[0] - left_hip[0])
        
        # Check if shoulder is steady (not swinging)
        if left_shoulder_hip_x_diff > 0.2 * width:
            form_issue = "Keep left shoulder stable"
            good_form_streak = max(0, good_form_streak - 1)
        else:
            form_issue = ""
            
        # Detect curl state
        if left_angle > 160 and left_stage == "up":
            left_stage = "down"
            feedback = "Left arm extended"
        elif left_angle < 50 and left_stage == "down":
            left_stage = "up"
            left_counter += 1
            feedback = f"Left curl complete: {left_counter}"
            if not form_issue:
                good_form_streak += 1
    
    # Process right arm curl
    if right_visible:
        # Check for proper form
        right_hip = (lm[24].x * width, lm[24].y * height)
        right_shoulder_hip_x_diff = abs(right_shoulder[0] - right_hip[0])
        
        # Check if shoulder is steady
        if right_shoulder_hip_x_diff > 0.2 * width:
            form_issue = "Keep right shoulder stable"
            good_form_streak = max(0, good_form_streak - 1)
        else:
            form_issue = ""
            
        # Detect curl state
        if right_angle > 160 and right_stage == "up":
            right_stage = "down"
            if not feedback:
                feedback = "Right arm extended"
        elif right_angle < 50 and right_stage == "down":
            right_stage = "up"
            right_counter += 1
            feedback = f"Right curl complete: {right_counter}"
            if not form_issue:
                good_form_streak += 1
    
    # Generate appropriate feedback based on form
    if not (left_visible or right_visible):
        form_feedback = "Position your arms in view of the camera"
    elif form_issue:
        form_feedback = form_issue
    elif 30 <= left_angle <= 160 or 30 <= right_angle <= 160:
        # Mid-motion feedback on form
        elbow_status = []
        
        # Check if elbows are too far from body
        if left_visible:
            left_hip_z = lm[23].z
            elbow_hip_z_diff = abs(left_elbow[2] - left_hip_z)
            if elbow_hip_z_diff > 0.1:
                elbow_status.append("Keep left elbow close to body")
        
        if right_visible:
            right_hip_z = lm[24].z
            elbow_hip_z_diff = abs(right_elbow[2] - right_hip_z)
            if elbow_hip_z_diff > 0.1:
                elbow_status.append("Keep right elbow close to body")
        
        if elbow_status:
            form_feedback = " & ".join(elbow_status)
        else:
            form_feedback = "Good form! Keep going"
    else:
        form_feedback = "Fully extend and curl arms"
    
    # Recognize and reward consistent good form
    if good_form_streak >= 5:
        form_feedback = "Excellent form! Keep it up!"
    
    return {
        "left_angle": round(left_angle, 2),
        "right_angle": round(right_angle, 2),
        "left_stage": left_stage,
        "right_stage": right_stage,
        "left_counter": left_counter,
        "right_counter": right_counter,
        "left_visible": left_visible,
        "right_visible": right_visible,
        "message": feedback,
        "form_feedback": form_feedback,
        "good_form_streak": good_form_streak
    }

def squat_counter(image, lm, width, height, stage, counter, good_form_streak):
    """
    Track squat exercise with detailed form correction feedback.
    """
    # Extract landmarks for both sides (for greater accuracy)
    # Left side
    left_hip = (lm[23].x * width, lm[23].y * height, lm[23].z)
    left_knee = (lm[25].x * width, lm[25].y * height, lm[25].z)
    left_ankle = (lm[27].x * width, lm[27].y * height, lm[27].z)
    
    # Right side
    right_hip = (lm[24].x * width, lm[24].y * height, lm[24].z)
    right_knee = (lm[26].x * width, lm[26].y * height, lm[26].z)
    right_ankle = (lm[28].x * width, lm[28].y * height, lm[28].z)
    
    # Shoulder points for back angle calculation
    left_shoulder = (lm[11].x * width, lm[11].y * height, lm[11].z)
    right_shoulder = (lm[12].x * width, lm[12].y * height, lm[12].z)
    
    # Calculate joint angles (using average of both sides for stability)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    knee_angle = (left_knee_angle + right_knee_angle) / 2
    
    # Calculate hip angle (lower back posture)
    # Use midpoints for more stable measurements
    mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                    (left_shoulder[1] + right_shoulder[1]) / 2, 
                    (left_shoulder[2] + right_shoulder[2]) / 2)
    mid_hip = ((left_hip[0] + right_hip[0]) / 2, 
               (left_hip[1] + right_hip[1]) / 2, 
               (left_hip[2] + right_hip[2]) / 2)
    mid_knee = ((left_knee[0] + right_knee[0]) / 2, 
                (left_knee[1] + right_knee[1]) / 2, 
                (left_knee[2] + right_knee[2]) / 2)
    
    # Back angle (should remain relatively straight)
    back_angle = calculate_angle(mid_shoulder, mid_hip, mid_knee)
    
    # Visualize angles
    visualize_angle(image, 
                   (left_hip[0], left_hip[1]), 
                   (left_knee[0], left_knee[1]), 
                   (left_ankle[0], left_ankle[1]), 
                   left_knee_angle, 
                   color=(255, 0, 0))
    
    visualize_angle(image, 
                   (right_hip[0], right_hip[1]), 
                   (right_knee[0], right_knee[1]), 
                   (right_ankle[0], right_ankle[1]), 
                   right_knee_angle, 
                   color=(0, 0, 255))
    
    visualize_angle(image,
                   (mid_shoulder[0], mid_shoulder[1]),
                   (mid_hip[0], mid_hip[1]),
                   (mid_knee[0], mid_knee[1]),
                   back_angle,
                   color=(0, 255, 0))
    
    # Check if landmarks are visible for reliable detection
    hip_knee_visible = all(lm[i].visibility > 0.7 for i in [23, 24, 25, 26])
    
    form_feedback = ""
    message = ""
    
    if hip_knee_visible:
        # Check for proper form
        
        # 1. Knee tracking over toes
        left_foot_index = (lm[31].x * width, lm[31].y * height)
        right_foot_index = (lm[32].x * width, lm[32].y * height)
        
        left_knee_x, right_knee_x = left_knee[0], right_knee[0]
        left_foot_x, right_foot_x = left_foot_index[0], right_foot_index[0]
        
        # Check if knees go too far beyond toes
        knee_over_toes = (abs(left_knee_x - left_foot_x) > 0.2 * width or 
                          abs(right_knee_x - right_foot_x) > 0.2 * width)
        
        # 2. Back alignment
        back_aligned = 70 <= back_angle <= 110
        
        # 3. Shoulder alignment (not leaning forward too much)
        shoulder_hip_vertical = abs(mid_shoulder[0] - mid_hip[0]) < 0.15 * width
        
        # Squat depth detection
        if knee_angle < 110 and stage == "up":
            stage = "down"
            message = "Good depth"
        elif knee_angle > 170 and stage == "down":
            stage = "up"
            counter += 1
            message = f"Squat complete: {counter}"
            
            # Check form only when completing a rep
            if not back_aligned:
                form_feedback = "Keep your back straight"
                good_form_streak = max(0, good_form_streak - 1)
            elif knee_over_toes:
                form_feedback = "Knees shouldn't go too far past toes"
                good_form_streak = max(0, good_form_streak - 1)
            elif not shoulder_hip_vertical:
                form_feedback = "Avoid leaning forward too much"
                good_form_streak = max(0, good_form_streak - 1)
            else:
                good_form_streak += 1
                form_feedback = "Good form!"
        
        # Continuous form feedback during rep
        if stage == "down":
            if not back_aligned:
                form_feedback = "Keep your back straight"
            elif knee_over_toes:
                form_feedback = "Align knees over ankles, not past toes"
            elif knee_angle > 110:  # Not deep enough
                form_feedback = "Lower for full range of motion"
            else:
                form_feedback = "Good depth, maintain form"
        elif stage == "up" and knee_angle < 170:
            form_feedback = "Stand up completely to finish rep"
    else:
        form_feedback = "Position your full body in view of the camera"
    
    # Recognize consistent good form
    if good_form_streak >= 3:
        form_feedback = "Excellent squat form! Keep it up!"
    
    # Prepare result
    return {
        "knee_angle": round(knee_angle, 2),
        "back_angle": round(back_angle, 2),
        "stage": stage,
        "counter": counter,
        "message": message,
        "form_feedback": form_feedback,
        "good_form_streak": good_form_streak
    }

def plank_hold(image, lm, width, height, duration, good_form_streak):
    """
    Track plank exercise with form correction feedback.
    """
    # Extract landmarks
    left_shoulder = (lm[11].x * width, lm[11].y * height, lm[11].z)
    right_shoulder = (lm[12].x * width, lm[12].y * height, lm[12].z)
    
    left_hip = (lm[23].x * width, lm[23].y * height, lm[23].z)
    right_hip = (lm[24].x * width, lm[24].y * height, lm[24].z)
    
    left_knee = (lm[25].x * width, lm[25].y * height, lm[25].z)
    right_knee = (lm[26].x * width, lm[26].y * height, lm[26].z)
    
    left_ankle = (lm[27].x * width, lm[27].y * height, lm[27].z)
    right_ankle = (lm[28].x * width, lm[28].y * height, lm[28].z)
    
    # Calculate body alignment angles
    # Mid points for more stable measurement
    mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                    (left_shoulder[1] + right_shoulder[1]) / 2, 
                    (left_shoulder[2] + right_shoulder[2]) / 2)
    
    mid_hip = ((left_hip[0] + right_hip[0]) / 2, 
               (left_hip[1] + right_hip[1]) / 2, 
               (left_hip[2] + right_hip[2]) / 2)
    
    mid_knee = ((left_knee[0] + right_knee[0]) / 2, 
                (left_knee[1] + right_knee[1]) / 2, 
                (left_knee[2] + right_knee[2]) / 2)
    
    mid_ankle = ((left_ankle[0] + right_ankle[0]) / 2, 
                 (left_ankle[1] + right_ankle[1]) / 2, 
                 (left_ankle[2] + right_ankle[2]) / 2)
    
    # Calculate alignment angles
    shoulder_hip_angle = calculate_angle(mid_shoulder, mid_hip, mid_knee)
    hip_knee_angle = calculate_angle(mid_hip, mid_knee, mid_ankle)
    
    # Visualize angles
    visualize_angle(image, 
                   (mid_shoulder[0], mid_shoulder[1]), 
                   (mid_hip[0], mid_hip[1]), 
                   (mid_knee[0], mid_knee[1]), 
                   shoulder_hip_angle, 
                   color=(0, 255, 0))
    
    visualize_angle(image, 
                   (mid_hip[0], mid_hip[1]), 
                   (mid_knee[0], mid_knee[1]), 
                   (mid_ankle[0], mid_ankle[1]), 
                   hip_knee_angle, 
                   color=(0, 0, 255))
    
    # Check if landmarks are visible
    body_visible = all(lm[i].visibility > 0.7 for i in [11, 12, 23, 24, 25, 26, 27, 28])
    
    # Form assessment
    form_feedback = ""
    
    if body_visible:
        # Ideal plank angles
        ideal_shoulder_hip_angle = 180  # Straight line from shoulders through hips
        ideal_hip_knee_angle = 180  # Straight line from hips through knees
        
        # Calculate deviation from ideal
        shoulder_hip_deviation = abs(shoulder_hip_angle - ideal_shoulder_hip_angle)
        hip_knee_deviation = abs(hip_knee_angle - ideal_hip_knee_angle)
        
        # Head position (should be neutral, looking down)
        nose = (lm[0].x * width, lm[0].y * height)
        neck = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                ((left_shoulder[1] + right_shoulder[1]) / 2) - 30)  # Approximation
        
        # Check various form issues
        if shoulder_hip_deviation > 15:
            if shoulder_hip_angle < 165:
                form_feedback = "Hips too low! Lift to align with shoulders"
                good_form_streak = max(0, good_form_streak - 1)
            else:
                form_feedback = "Hips too high! Lower to align with shoulders"
                good_form_streak = max(0, good_form_streak - 1)
        elif hip_knee_deviation > 15:
            form_feedback = "Straighten your legs for proper alignment"
            good_form_streak = max(0, good_form_streak - 1)
        elif abs(mid_shoulder[0] - mid_hip[0]) > 0.1 * width:
            form_feedback = "Body not aligned horizontally, adjust position"
            good_form_streak = max(0, good_form_streak - 1)
        else:
            good_form_streak += 1
            form_feedback = "Great plank form!"
            
        # Advanced feedback with holding time
        if duration > 30 and duration % 10 < 0.1:  # Provide encouragement at interval
            form_feedback = f"Excellent! {round(duration)}s - Keep breathing!"
            
        return {
        "shoulder_hip_angle": round(shoulder_hip_angle, 2),
        "hip_knee_angle": round(hip_knee_angle, 2),
        "form_feedback": form_feedback,
        "good_form_streak": good_form_streak,
        "message": f"Hold time: {round(duration, 1)}s"
    }